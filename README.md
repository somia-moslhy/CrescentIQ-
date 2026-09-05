# CrescentIQ

**Hijri-aware demand forecasting for MENA retail & manufacturing**

CrescentIQ is a demand forecasting engine that adds a Hijri-calendar feature layer to a classical ML pipeline, and measures — with numbers, not assumptions — how much it actually improves forecast accuracy around Ramadan and the two Eids.

## The Problem

Enterprise forecasting tools (SAP, Oracle, and even standard SARIMA/Prophet setups) assume Gregorian-calendar seasonality. But Ramadan and Eid follow the lunar Hijri calendar, which shifts ~10-11 days earlier every Gregorian year. A model trained to expect a demand spike in a fixed Gregorian month will predict it in the wrong week the very next year — leading to real overstock/stockout costs for retailers and manufacturers across Egypt and Saudi Arabia. No mainstream forecasting tool accounts for this.

## The Approach

- **Data**: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) (Kaggle) — ~421K rows, mixed numerical/categorical columns, real missing values (not pre-cleaned).
- **Feature engineering**: `days_to`/`days_since` Ramadan, Eid al-Fitr, and Eid al-Adha, computed via the `hijri-converter` library (Umm al-Qura calendar) — not hand-typed dates.
- **Modeling**: Classical ML only (no deep learning). Three tree-based models compared on identical features and a time-based train/test split: **RandomForest** (bagging), **XGBoost** and **LightGBM** (gradient boosting).
- **Isolating the Hijri effect**: the winning model is evaluated twice — with and without the Hijri features — to measure their contribution in isolation, rather than confounding it with a change of algorithm.
- **Everything runs inside an `sklearn.Pipeline`**: encoding and statistical imputation are fit on train data only, avoiding leakage.

## Results

| Model | Overall RMSE | Overall R² |
|---|---|---|
| **LightGBM** (best) | lowest | ~0.91 |
| XGBoost | higher | lower |
| RandomForest | highest | lowest |

Adding Hijri features improved Ramadan/Eid-window RMSE by **~3% overall** — concentrated much more heavily in specific departments (see `results/improvement_by_dept.csv` and the app's Analysis page for the full breakdown).

## Live Demo

- Streamlit app: [https://x3sjmqf8kgfdmcw3aeljts.streamlit.app/](https://x3sjmqf8kgfdmcw3aeljts.streamlit.app/)
- Explainer video: [https://notebook.google.com/notebook/393b45b6-1c37-4cc3-9202-78da3e1cb611/artifact/bb9055b2-d562-4d4c-819b-7784d1ee579d?utm_source=nlm_web_share&utm_medium=google_oo&utm_campaign=art_share_1&utm_content=&utm_smc=nlm_web_share_google_oo_art_share_1_](https://notebook.google.com/notebook/393b45b6-1c37-4cc3-9202-78da3e1cb611/artifact/bb9055b2-d562-4d4c-819b-7784d1ee579d?utm_source=nlm_web_share&utm_medium=google_oo&utm_campaign=art_share_1&utm_content=&utm_smc=nlm_web_share_google_oo_art_share_1_)
- Presentation: [https://gamma.app/docs/CrescentIQ-Hijri-Aware-Demand-Forecasting-0lylm2c496mmxl0](https://gamma.app/docs/CrescentIQ-Hijri-Aware-Demand-Forecasting-0lylm2c496mmxl0)

## Tech Stack

`pandas` · `scikit-learn` · `LightGBM` · `XGBoost` · `hijri-converter` · `Plotly` · `Streamlit`

## ⚖️ License & Copyright

**CrescentIQ** is the original intellectual property of [Somia Moslhy Afify]. 

This project is licensed under the  MIT License - see the [LICENSE](LICENSE) file for details.

**Academic & Professional Citation:**
If you use this methodology, code, or the concept of integrating Hijri calendar proximity features for demand forecasting in your own projects, research, or commercial applications, you are required to provide proper attribution to the original author.

*© 2026 Somia Moslhy Afify. All Rights Reserved.*
