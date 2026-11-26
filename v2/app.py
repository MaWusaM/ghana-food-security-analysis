"""
Ghana Regional Food Security & Climate Stress Simulator (v2)
============================================================
Interactive Streamlit decision-support app powered by the expanding-window
HistGradientBoosting & Random Forest ensembles saved in `models/ghana_food_v2.joblib`.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration & Cached Artifact Loaders
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ghana Food Security Simulator v2",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/ghana_food_v2.joblib")
DATA_PATH = Path("data/processed/ghana_model_features.parquet")


@st.cache_resource
def load_model_bundle(path: Path) -> dict:
    """Loads the serialized regressor, classifier, feature list, and metadata."""
    return joblib.load(path)


@st.cache_data
def load_panel_data(path: Path) -> pd.DataFrame:
    """Loads the 2012–2024 leak-free regional modeling panel."""
    return pd.read_parquet(path).sort_values(["region", "year"]).reset_index(drop=True)


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error(
        "Missing model or dataset artifacts. Confirm `models/ghana_food_v2.joblib` "
        "and `data/processed/ghana_model_features.parquet` exist in your project directory."
    )
    st.stop()

bundle = load_model_bundle(MODEL_PATH)
df = load_panel_data(DATA_PATH)

reg_model = bundle["regressor"]
cls_model = bundle["classifier"]
FEATURE_COLS = bundle["feature_cols"]
KCAL_THRESHOLD = float(bundle.get("caloric_threshold_kcal", 2100.0))

# Precompute regional historical rainfall mean & std for consistent z-score updates
region_rain_stats = (
    df.groupby("region")["annual_rainfall_mm"]
    .agg(mean_rain="mean", std_rain="std")
    .assign(std_rain=lambda d: d["std_rain"].replace(0, 1.0))
)


def apply_what_if_shocks(
    base_rows: pd.DataFrame,
    rain_anom_delta_pct: float,
    dry_dekad_delta: int,
    price_shock_pct: float,
    shannon_delta: float,
    cereal_share_delta: float,
) -> pd.DataFrame:
    """
    Applies user-selected climate, price, and cropping-system shocks to a copy
    of the baseline feature rows while keeping engineered cross-features consistent.
    """
    sim = base_rows.copy()

    # 1. Growing-season rainfall anomaly & proportional annual rainfall adjustment
    sim["rain_anomaly_t"] = (sim["rain_anomaly_t"] + rain_anom_delta_pct).clip(40.0, 160.0)
    rain_scale = float(np.clip(1.0 + rain_anom_delta_pct / 100.0, 0.50, 1.50))
    sim["annual_rainfall_mm"] = sim["annual_rainfall_mm"] * rain_scale

    # Recompute regional rainfall z-score and 2-year compound drought index
    means = sim["region"].map(region_rain_stats["mean_rain"])
    stds = sim["region"].map(region_rain_stats["std_rain"])
    sim["rain_zscore_t"] = (sim["annual_rainfall_mm"] - means) / stds
    sim["compound_2yr_rain_anom"] = 0.6 * sim["rain_anomaly_t"] + 0.4 * sim["lag1_rain_anomaly"]

    # 2. Dry dekads (10-day dry spells during the cropping season)
    sim["dry_dekads_count"] = (sim["dry_dekads_count"] + dry_dekad_delta).clip(0, 24)

    # 3. Rural wholesale staple price shock
    price_mult = float(np.clip(1.0 + price_shock_pct / 100.0, 0.50, 2.50))
    sim["lag1_wholesale_price_ghs"] = sim["lag1_wholesale_price_ghs"] * price_mult
    sim["lag1_price_yoy_pct"] = sim["lag1_price_yoy_pct"] + price_shock_pct

    # 4. Agro-ecological resilience levers: crop diversification & cereal vs. root mix
    sim["lag1_crop_shannon_idx"] = (sim["lag1_crop_shannon_idx"] + shannon_delta).clip(0.0, 2.1)
    sim["lag1_cereal_kcal_share"] = (sim["lag1_cereal_kcal_share"] + cereal_share_delta).clip(0.05, 0.95)

    return sim


# ---------------------------------------------------------------------------
# Sidebar: Scenario & "What-If" Policy Controls
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ Scenario & Policy Controls")

regions_sorted = sorted(df["region"].unique().tolist())
years_sorted = sorted(df["year"].unique().tolist(), reverse=True)

selected_region = st.sidebar.selectbox("Target Legacy Region", regions_sorted, index=0)
selected_year = st.sidebar.selectbox("Reference Baseline Year", years_sorted, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Climate & Market Shocks")

rain_anom_delta = st.sidebar.slider(
    "Growing-Season Rainfall Anomaly Shift (% pts)",
    min_value=-40,
    max_value=30,
    value=0,
    step=5,
    help="Negative values simulate drought stress; positive values simulate wetter seasons.",
)

dry_dekad_delta = st.sidebar.slider(
    "Additional Dry Dekads (10-day dry spells)",
    min_value=-4,
    max_value=8,
    value=0,
    step=1,
)

price_shock_pct = st.sidebar.slider(
    "Wholesale Staple Price Shock (%)",
    min_value=-20,
    max_value=60,
    value=0,
    step=5,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Agro-Ecological Resilience Levers")

shannon_delta = st.sidebar.slider(
    "Crop Diversity Shift (Shannon Index)",
    min_value=-0.40,
    max_value=0.40,
    value=0.00,
    step=0.05,
    help="Higher Shannon Index reflects broader multi-crop diversification.",
)

cereal_share_delta = st.sidebar.slider(
    "Cereal vs. Root/Tuber Share Shift",
    min_value=-0.20,
    max_value=0.20,
    value=0.00,
    step=0.05,
)

# ---------------------------------------------------------------------------
# Header & Baseline vs. Simulated Inference
# ---------------------------------------------------------------------------
st.title("🌾 Ghana Regional Food Security & Climate Stress Simulator (v2)")
st.caption(
    f"Regressor: **{bundle['regressor_name']}** | "
    f"Deficit Classifier: **{bundle['classifier_name']}** | "
    f"FAO/WFP Caloric Threshold: **{KCAL_THRESHOLD:,.0f} kcal/person/day**"
)

year_slice = df[df["year"] == selected_year].copy().reset_index(drop=True)
sim_year_slice = apply_what_if_shocks(
    year_slice,
    rain_anom_delta,
    dry_dekad_delta,
    price_shock_pct,
    shannon_delta,
    cereal_share_delta,
)

year_slice["pred_kcal_base"] = reg_model.predict(year_slice[FEATURE_COLS])
year_slice["risk_prob_base"] = cls_model.predict_proba(year_slice[FEATURE_COLS])[:, 1]

sim_year_slice["pred_kcal_sim"] = reg_model.predict(sim_year_slice[FEATURE_COLS])
sim_year_slice["risk_prob_sim"] = cls_model.predict_proba(sim_year_slice[FEATURE_COLS])[:, 1]

reg_base = year_slice[year_slice["region"] == selected_region].iloc[0]
reg_sim = sim_year_slice[sim_year_slice["region"] == selected_region].iloc[0]

# Top-level KPI Metrics
k1, k2, k3, k4 = st.columns(4)
kcal_diff = reg_sim["pred_kcal_sim"] - reg_base["pred_kcal_base"]
prob_diff = (reg_sim["risk_prob_sim"] - reg_base["risk_prob_base"]) * 100.0

k1.metric(
    label=f"{selected_region} Observed ({selected_year})",
    value=f"{reg_base['staple_kcal_per_capita_day']:,.0f} kcal/day",
    delta=f"{reg_base['staple_kcal_per_capita_day'] - KCAL_THRESHOLD:+,.0f} vs threshold",
)
k2.metric(
    label="Model Baseline Prediction",
    value=f"{reg_base['pred_kcal_base']:,.0f} kcal/day",
    delta=f"Deficit Risk: {reg_base['risk_prob_base']*100:.1f}%",
    delta_color="off",
)
k3.metric(
    label="Simulated Scenario Output",
    value=f"{reg_sim['pred_kcal_sim']:,.0f} kcal/day",
    delta=f"{kcal_diff:+,.0f} kcal vs baseline",
)
k4.metric(
    label="Simulated Deficit Probability",
    value=f"{reg_sim['risk_prob_sim']*100:.1f}%",
    delta=f"{prob_diff:+.1f} percentage pts",
    delta_color="inverse",
)

# ---------------------------------------------------------------------------
# Main Dashboard Tabs
# ---------------------------------------------------------------------------
tab_sim, tab_nat, tab_audit = st.tabs(
    [
        "📈 Regional Trajectory & What-If",
        "🗺️ 10-Region National Stress Test",
        "🔍 Model Explainability & Backtest",
    ]
)

with tab_sim:
    reg_hist = df[df["region"] == selected_region].sort_values("year")

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=reg_hist["year"],
            y=reg_hist["staple_kcal_per_capita_day"],
            mode="lines+markers",
            name="Observed Staple Supply (kcal/day)",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig_hist.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[reg_sim["pred_kcal_sim"]],
            mode="markers",
            name="Simulated What-If Scenario",
            marker=dict(color="#d62728", size=14, symbol="diamond"),
        )
    )
    fig_hist.add_hline(
        y=KCAL_THRESHOLD,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text=f"FAO/WFP Threshold ({KCAL_THRESHOLD:,.0f} kcal/day)",
    )
    fig_hist.update_layout(
        title=f"{selected_region}: Historical Caloric Supply vs. Simulated Scenario ({selected_year})",
        xaxis_title="Year",
        yaxis_title="Staple Availability (kcal / person / day)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Inspect Baseline vs. Shocked Feature Vector"):
        comp_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLS,
                "Baseline Value": [round(float(reg_base[c]), 3) for c in FEATURE_COLS],
                "Simulated Value": [round(float(reg_sim[c]), 3) for c in FEATURE_COLS],
            }
        )
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

with tab_nat:
    nat_comp = pd.DataFrame(
        {
            "Region": year_slice["region"],
            "Observed_kcal": year_slice["staple_kcal_per_capita_day"].round(0),
            "Baseline_Pred_kcal": year_slice["pred_kcal_base"].round(0),
            "Simulated_kcal": sim_year_slice["pred_kcal_sim"].round(0),
            "Delta_kcal": (sim_year_slice["pred_kcal_sim"] - year_slice["pred_kcal_base"]).round(0),
            "Simulated_Risk_Prob": (sim_year_slice["risk_prob_sim"] * 100).round(1),
            "Status": np.where(
                sim_year_slice["pred_kcal_sim"] < KCAL_THRESHOLD,
                "⚠️ Deficit Risk (<2,100)",
                "✅ Adequate Supply",
            ),
        }
    ).sort_values("Simulated_kcal")

    fig_bar = px.bar(
        nat_comp,
        x="Simulated_kcal",
        y="Region",
        color="Status",
        orientation="h",
        color_discrete_map={
            "⚠️ Deficit Risk (<2,100)": "#d62728",
            "✅ Adequate Supply": "#2ca02c",
        },
        title=f"Simulated Daily Staple Calories Across All 10 Legacy Regions ({selected_year})",
        labels={"Simulated_kcal": "Predicted Supply (kcal/person/day)"},
        height=440,
    )
    fig_bar.add_vline(x=KCAL_THRESHOLD, line_dash="dash", line_color="#ff7f0e")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(nat_comp, hide_index=True, use_container_width=True)

with tab_audit:
    c_left, c_right = st.columns([1.2, 1.0])

    with c_left:
        st.subheader("Top Model Drivers (Permutation R² Drop)")
        imp_records = bundle.get("feature_importance", [])
        if imp_records:
            imp_df = pd.DataFrame(imp_records).head(10).iloc[::-1]
            fig_imp = px.bar(
                imp_df,
                x="Importance_R2_Drop",
                y="Feature",
                orientation="h",
                title=f"Feature Importance ({bundle['regressor_name']})",
                labels={"Importance_R2_Drop": "Mean R² Drop When Shuffled"},
                height=390,
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Run the Week 2 permutation importance patch to populate driver rankings.")

    with c_right:
        st.subheader("Expanding-Window Backtest (2020–2024)")
        st.markdown("**Regression Out-of-Sample Metrics (`staple_kcal_per_capita_day`)**")
        st.dataframe(pd.DataFrame(bundle["regression_metrics"]), hide_index=True, use_container_width=True)

        st.markdown("**Classification Out-of-Sample Metrics (`deficit_risk_flag`)**")
        st.dataframe(pd.DataFrame(bundle["classification_metrics"]), hide_index=True, use_container_width=True)
