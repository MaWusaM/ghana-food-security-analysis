"""
src/ingest.py
=============
Week 1 Data Ingestion Pipeline for the Ghana Food Security Risk Simulator (v2).

Responsibilities:
1. Download and cache subnational CHIRPS rainfall data (10-day dekadal observations)
   from the Humanitarian Data Exchange (HDX / WFP).
2. Aggregate dekadal rainfall and drought-anomaly metrics to the Region x Year level.
3. Pull annual national macro-agricultural covariates (fertilizer use, cereal yields,
   population growth, inflation) from the World Bank API via `wbdata`.
4. Generate a structured CSV template for Ghana MoFA SRID regional staple crop data
   (maize, rice, cassava, yam, millet, sorghum) ready for PDF table extraction.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import wbdata

# ---------------------------------------------------------------------------
# 1. Directory & Constant Configuration
# ---------------------------------------------------------------------------
# Define local storage paths for raw downloaded files and cleaned Parquet outputs
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Direct download endpoint for WFP/HDX Ghana subnational rainfall estimates (CHIRPS)
HDX_RAINFALL_URL = (
    "https://data.humdata.org/dataset/187fc57e-07af-43d9-b412-6aaba91a3209/"
    "resource/23e0d7d8-2f33-4844-aaff-d056d26b385a/download/gha-rainfall-subnat-full.csv"
)

# Canonical 16 administrative regions of Ghana (aligned with post-2019 boundaries)
GHANA_REGIONS_16 = [
    "Ahafo", "Ashanti", "Bono", "Bono East", "Central", "Eastern",
    "Greater Accra", "North East", "Northern", "Oti", "Savannah",
    "Upper East", "Upper West", "Volta", "Western", "Western North"
]


# ---------------------------------------------------------------------------
# 2. Subnational Rainfall Ingestion (HDX / WFP CHIRPS)
# ---------------------------------------------------------------------------
def fetch_hdx_rainfall(force_download: bool = False) -> pd.DataFrame:
    """
    Downloads and aggregates HDX/WFP subnational CHIRPS rainfall to Region x Year.

    Parameters
    ----------
    force_download : bool, default False
        If True, re-downloads the CSV from HDX even if a cached copy exists locally.

    Returns
    -------
    pd.DataFrame
        Aggregated annual and growing-season rainfall features keyed by `region_key` and `year`.
    """
    raw_csv = RAW_DIR / "gha-rainfall-subnat-full.csv"

    # Cache the raw ~43 MB CSV locally so subsequent script runs execute in seconds
    if not raw_csv.exists() or force_download:
        print(f"Downloading HDX subnational rainfall data to {raw_csv}...")
        resp = requests.get(HDX_RAINFALL_URL, timeout=60)
        resp.raise_for_status()
        raw_csv.write_bytes(resp.content)
        print("Download complete.")
    else:
        print(f"Loading cached HDX rainfall from {raw_csv}...")

    # HDX datasets frequently include an HXL metadata tag row on line 2 (starting with '#').
    # Inspect the first two lines so we skip the HXL tag row without dropping real data.
    with open(raw_csv, "r", encoding="utf-8") as f:
        _ = f.readline()
        second_line = f.readline()
    skiprows = [1] if second_line.startswith("#") else None

    df = pd.read_csv(raw_csv, skiprows=skiprows, low_memory=False)

    # Parse observation dates and extract calendar year and month for seasonal filtering
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Detect whether the file already contains an Admin-1 region name column.
    # If only Admin-2 district P-codes (e.g., 'GH0101') are provided, slice the first
    # 4 characters ('GH01') to group observations at the Admin-1 regional level.
    admin1_col = next((c for c in ["ADM1_NAME", "adm1_name", "Region"] if c in df.columns), None)
    if admin1_col is None:
        pcode_col = next(c for c in ["ADM2_PCODE", "adm2_pcode", "ADM1_PCODE"] if c in df.columns)
        df["adm1_pcode"] = df[pcode_col].astype(str).str[:4]
        group_col = "adm1_pcode"
    else:
        group_col = admin1_col

    # Filter to 2000–2025 to align with modern MoFA agricultural reports and World Bank series
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)].copy()

    # Define Ghana's core agricultural rainy window (April–October), which spans both
    # the southern bimodal major/minor seasons and the northern unimodal rainy season.
    growing_season = df[df["month"].between(4, 10)]

    # Aggregate full-year rainfall totals and drought-stress indicators per region:
    # - rfh: 10-day (dekadal) rainfall total in millimeters
    # - rfq: 10-day rainfall anomaly (% of long-term average; <100% indicates drier than normal)
    # - r3q: 3-month rolling rainfall anomaly (% of long-term average)
    annual_rain = (
        df.groupby([group_col, "year"], as_index=False)
        .agg(
            annual_rainfall_mm=("rfh", "sum"),
            mean_10d_anomaly_pct=("rfq", "mean"),
            min_3m_anomaly_pct=("r3q", "min"),  # Captures the sharpest 3-month drought shock
            dry_dekads_count=("rfq", lambda s: (s < 75.0).sum()),  # Count of severe dry 10-day periods
        )
    )

    # Aggregate rainfall specifically during the April–October crop growing window
    season_rain = (
        growing_season.groupby([group_col, "year"], as_index=False)
        .agg(
            growing_season_rain_mm=("rfh", "sum"),
            growing_season_anomaly_pct=("rfq", "mean"),
        )
    )

    # Merge annual and growing-season metrics into a single Region x Year panel
    rain_panel = pd.merge(annual_rain, season_rain, on=[group_col, "year"], how="left")
    rain_panel.rename(columns={group_col: "region_key"}, inplace=True)
    return rain_panel


# ---------------------------------------------------------------------------
# 3. National Macro & Agricultural Input Covariates (World Bank API)
# ---------------------------------------------------------------------------
def fetch_worldbank_macro() -> pd.DataFrame:
    """
    Pulls Ghana national macro-agricultural covariates via `wbdata`.

    Returns
    -------
    pd.DataFrame
        Annual macro features for Ghana (2000–2025) to join onto regional rows.
    """
    print("Fetching World Bank macro indicators for Ghana (GHA)...")

    # Map World Bank indicator codes to readable column names
    indicators = {
        "AG.CON.FERT.ZS": "fertilizer_kg_per_ha",       # Fertilizer consumption (kg/ha of arable land)
        "AG.YLD.CREL.KG": "national_cereal_yield_kg_ha", # National cereal yield benchmark (kg/ha)
        "SP.POP.GROW": "pop_growth_pct",                 # Annual population growth rate (%)
        "SP.POP.TOTL": "national_population",            # Total national population
        "SN.ITK.DEFC.ZS": "undernourishment_pct",        # Prevalence of undernourishment (% of pop)
        "FP.CPI.TOTL.ZG": "inflation_cpi_pct",           # Consumer price inflation (annual %)
    }

    df_wb = wbdata.get_dataframe(indicators, country="GHA").reset_index()

    # Convert date string to integer year, filter to 2000–2025, and sort chronologically
    df_wb["year"] = pd.to_datetime(df_wb["date"]).dt.year
    df_wb = df_wb[(df_wb["year"] >= 2000) & (df_wb["year"] <= 2025)].sort_values("year")
    df_wb.drop(columns=["date"], inplace=True)

    # Forward-fill occasional trailing reporting lags in World Bank indicators (e.g., 2024–2025)
    df_wb = df_wb.ffill()
    return df_wb


# ---------------------------------------------------------------------------
# 4. MoFA Regional Staple Production Template Initialization
# ---------------------------------------------------------------------------
def init_mofa_crop_template() -> Path:
    """
    Creates a structured Region x Year CSV template for Ghana MoFA SRID
    staple production figures (to be populated via `pdfplumber` or manual entry).

    Returns
    -------
    Path
        File path to the generated template CSV.
    """
    template_path = RAW_DIR / "mofa_regional_staples_template.csv"

    # Only generate the template if it does not already exist so manual/scripted edits aren't overwritten
    if not template_path.exists():
        rows = []
        for year in range(2015, 2025):
            for region in GHANA_REGIONS_16:
                rows.append({
                    "region": region,
                    "year": year,
                    "regional_pop": np.nan,          # Regional population headcount
                    "maize_mt": np.nan,              # Production in Metric Tons (MT)
                    "rice_mt": np.nan,               # Paddy/milled rice production (MT)
                    "cassava_mt": np.nan,            # Cassava production (MT)
                    "yam_mt": np.nan,                # Yam production (MT)
                    "millet_mt": np.nan,             # Millet production (MT)
                    "sorghum_mt": np.nan,            # Sorghum production (MT)
                    "total_cropped_area_ha": np.nan, # Total cultivated staple area (Hectares)
                })
        pd.DataFrame(rows).to_csv(template_path, index=False)
        print(f"Created MoFA regional crop entry template at {template_path}")
    return template_path


# ---------------------------------------------------------------------------
# 5. Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Execute all ingestion steps
    rain_df = fetch_hdx_rainfall()
    wb_df = fetch_worldbank_macro()
    init_mofa_crop_template()

    # Export cleaned intermediate datasets to Parquet for fast downstream loading
    rain_df.to_parquet(PROCESSED_DIR / "ghana_subnat_rainfall.parquet", index=False)
    wb_df.to_parquet(PROCESSED_DIR / "ghana_wb_macro.parquet", index=False)

    # Print quick diagnostic summary to verify row counts, year coverage, and region keys
    print("\n--- Pipeline Check ---")
    print(f"Rainfall panel shape: {rain_df.shape} | Years: {rain_df['year'].min()}–{rain_df['year'].max()}")
    print(f"Unique region keys in rainfall: {rain_df['region_key'].unique().tolist()}")
    print(f"World Bank macro shape: {wb_df.shape}")
    print("\nSample rainfall rows:")
    print(rain_df.head())