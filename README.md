# Nifty 50 Market Intelligence Terminal

A modular Streamlit research application with two deliberately independent workstreams:

1. **Daily Nifty 50 forecasting** — OHLCV feature engineering, technical indicators, chronological ML evaluation, direction classification, exploratory clustering, risk diagnostics and forecast gating.
2. **Monthly SIP Shield research** — separate monthly SIP/FII/DII data preparation and an OLS interaction model that requires a genuine monthly Nifty return outcome.

## Important result from the supplied workbook

The inspected workbook has eight sheets. Its `Daily Nifty Model Data` sheet contains **22 sessions from 14-Oct-2026 to 13-Nov-2026**. This is not enough history for defensible model fitting, walk-forward evaluation or forecast calibration. The application therefore blocks daily supervised training until at least 80 labelled rows are available. This is a conservative software gate, not a universal statistical guarantee.

The `Monthly SIP FII DII` sheet contains **130 monthly observations from Jan-2016 to Oct-2026**, with **5 missing SIP inflow values**. The companion `Monthly Derived Vars` sheet contains 130 rows, with 2 missing prior-month return values and 4 missing 3-month volatility values. Critically, the workbook does **not** contain a current-month Nifty return outcome. `Previous Month Nifty Return (%)` is not silently substituted as the dependent variable. The SIP Shield hypothesis regression remains disabled until a proper monthly Nifty return series is supplied and aligned.

The full external VIX/S&P 500 sheet has 2,824 dates from Jan-2016 to Nov-2026. It contains 99 missing S&P 500 close values and 133 missing India VIX close values (100 and 134 missing calculated returns respectively). Missing observations are retained; no zero-fill or fabricated market observations are applied.

## Run on Windows

1. Install **Python 3.10 or 3.11** from python.org. During installation, tick **Add Python to PATH**.
2. Extract the ZIP to a normal folder, e.g. `C:\Projects\nifty50_market_intelligence`.
3. Double-click `run_app.bat`. It creates a virtual environment, installs dependencies and launches Streamlit.
4. In the browser, upload `Nifty50_Daily_and_Monthly_Dataset_V2.xlsx` using the sidebar workbook uploader.

Manual alternative in PowerShell / Command Prompt:

```bash
cd path\to\nifty50_market_intelligence
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Updating data

- Upload an Excel workbook with sheets named `Daily Nifty Model Data`, `Monthly SIP FII DII`, `Monthly Derived Vars`, and optionally `Full Daily VIX SP500`.
- Or upload a daily CSV with `Date`, `Nifty_Open`, `Nifty_High`, `Nifty_Low`, `Nifty_Close` and optional volume/VIX/S&P fields.
- Monthly research can also be uploaded as a separate CSV. Keep one row per calendar month; do not repeat monthly values on daily rows.
- To enable the SIP Shield regression, include a correctly aligned current-month outcome column named `Nifty Monthly Return (%)` (or one of the accepted aliases in `modules/sip_research.py`). Include provenance, units, and observation timing in your source notes.
- Original uploaded files are not altered or bundled in the ZIP. Missing values remain missing and are handled only inside training pipelines when model fitting is enabled.

## Validation and leakage controls

- Chronological 80/20 holdout; no random split or shuffle.
- TimeSeriesSplit expanding-window folds for regression cross-validation.
- Scikit-learn Pipeline keeps median imputation and scaling inside each fitted fold/model.
- Next-session target is shifted one session forward; target fields are excluded from predictors.
- A training-mean baseline is reported with regression metrics.
- Negative holdout R² is retained as-is.
- The final holdout is not used for fitting or feature selection.
- For small datasets, training is disabled rather than creating synthetic market data.

## Implemented algorithms

Regression: Linear Regression, Ridge, Lasso, Random Forest, Extra Trees, HistGradientBoosting.

Classification: Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting.

Unsupervised: K-Means and PCA (exploratory; sample-size cautions are shown).

Econometrics: statsmodels OLS with SIP × FII-selling interaction, HC3 robust covariance, VIF and Breusch–Pagan test when a valid dependent variable exists.

XGBoost and SHAP are intentionally optional and not required. XGBoost compatibility varies by Python/platform; SHAP is not a hard dependency. Permutation importance and linear coefficients provide dependency-light explainability.

## GitHub upload

1. Create a new private or public GitHub repository.
2. Extract this ZIP and upload the project files (do not upload confidential data or `.venv`).
3. Commit `app.py`, `modules/`, `tests/`, `requirements.txt`, `README.md` and `run_app.bat`.
4. Keep credentials out of source control. This project does not require API keys.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. In Streamlit Community Cloud, create an app and select the repository, branch and `app.py`.
3. Deploy. Upload your Excel workbook through the app sidebar after launch. Do not commit large or sensitive datasets to the repository.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

See `TEST_REPORT.md` and `reports/data_audit_summary.md` for the executed local checks and the source workbook audit.

## Disclaimer

For education and research only; not investment advice. No forecast is guaranteed. Statistical association is not causation. Verify source provenance, units, timestamp availability and corporate/market calendar alignment before relying on any result.
