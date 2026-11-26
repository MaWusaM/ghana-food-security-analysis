# Ghana Regional Food Security & Climate Stress Simulator (v2)

An end-to-end spatial-temporal machine learning pipeline and interactive Streamlit decision-support simulator forecasting **regional staple caloric availability (`kcal/person/day`)** and **caloric deficit risk (`< 2,100 kcal/day` FAO/WFP threshold)** across Ghana's 10 legacy administrative regions (`2012-2024`).

---

## Project Architecture & Data Pipeline

1. **Subnational Climate & Macro Ingestion:**
   - Harmonizes subnational CHIRPS rainfall metrics (`annual_rainfall_mm`, `growing_season_anomaly_pct`, `dry_dekads_count`) across Ghana's 10 legacy administrative regions (`GH02`-`GH15`), rolling up post-2019 regional splits (`Ahafo`, `Bono`, `Bono East`, `North East`, `Savannah`, `Oti`, `Western North`) back to consistent legacy boundaries.
   - Integrates World Bank macroeconomic & demographic series and MoFA SRID *Facts & Figures* (`2019` & `2024` editions) national crop production (`2010-2024`), regional crop shares, and rural wholesale staple prices.

2. **Leak-Free Temporal Feature Engineering (`data/processed/ghana_model_features.parquet`):**
   - **Lagged Supply Momentum:** `lag1_kcal_per_capita`, `lag1_kcal_yoy_pct`, `lag1_pop_density_proxy`.
   - **Compound Climate Shocks:** `rain_anomaly_t`, `lag1_rain_anomaly`, `compound_2yr_rain_anom` (`0.6 * anom_t + 0.4 * anom_{t-1}`), `rain_zscore_t`, and `dry_dekads_count`.
   - **Agro-Ecological Resilience & Market Stress:** Lagged crop portfolio Shannon Diversity Index (`lag1_crop_shannon_idx`), cereal vs. root/tuber caloric share (`lag1_cereal_kcal_share`), and rural wholesale price shocks (`lag1_wholesale_price_ghs`, `lag1_price_yoy_pct`).

3. **Panel-Aware Expanding-Window Validation (`2020-2024` Holdout):**
   - Evaluates models strictly forward in time (`Train <= t-1` -> `Test year t`, keeping all 10 regions of each test year together to prevent cross-region macro leakage).

| Task | Selected Model | Out-of-Sample Metric (2020-2024) |
| :--- | :--- | :--- |
| **Caloric Supply Regression** (`staple_kcal_per_capita_day`) | `HistGradientBoostingRegressor` | **$R^2 = 0.900$** \| MAE = `719.5 kcal/day` \| RMSE = `977.1 kcal/day` |
| **Deficit Risk Classification** (`< 2,100 kcal/day`) | `RandomForestClassifier` (balanced) | **ROC-AUC = `1.000`** \| **PR-AUC = `1.000`** \| **F1 = `1.000`** |

*(Note: Modeled caloric estimates reflect regional domestic staple production converted via FAO/WFP energy densities after an assumed 85% net edible/post-harvest availability factor, prior to inter-regional trade and commercial imports.)*

---

## Interactive Streamlit Dashboard (`app.py`)

- **Scenario & Policy Controls:** Adjust growing-season rainfall anomalies (`-40%` to `+30%`), dry dekads (`-4` to `+8`), wholesale staple price shocks (`-20%` to `+60%`), and cropping-system diversification levers (`Shannon Index`, cereal share).
- **Regional Trajectory & What-If:** Compares observed historical caloric trajectories (`2012-2024`) against simulated shock outcomes and exposes the baseline vs. shocked feature vector.
- **10-Region National Stress Test:** Applies active climate/price shocks across all 10 regions simultaneously to identify regions tipping below the `2,100 kcal/person/day` threshold.
- **Model Explainability & Backtest Audit:** Displays permutation importance rankings (`R^2` drop) and out-of-sample backtest scorecards.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive simulator
streamlit run app.py
```

## Repository Structure

```text
v2/
├── app.py                                      # Interactive Streamlit dashboard
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation & validation summary
├── models/
│   └── ghana_food_v2.joblib                    # Serialized regressor, classifier & metadata
├── data/
│   └── processed/
│       ├── ghana_subnat_rainfall.parquet       # Harmonized 10-region CHIRPS rainfall
│       ├── ghana_wb_macro.parquet              # World Bank macro indicators
│       ├── ghana_food_panel.parquet            # Master 10-region x 15-year panel
│       └── ghana_model_features.parquet        # Leak-free modeling feature matrix
└── reports/
    └── figures/
        └── shap_caloric_drivers.png            # Permutation feature importance plot
```
