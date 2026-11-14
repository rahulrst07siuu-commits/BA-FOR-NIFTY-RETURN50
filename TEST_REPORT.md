# Test and execution report

**Execution date:** 14-Nov-2026 (environment local date)

## Results summary

| Check | Result | Evidence / limitation |
|---|---|---|
| Python compilation | PASS | `python -m compileall -q .` completed without syntax errors. |
| Core dependency imports | PASS | pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, Plotly 6.5.2, statsmodels 0.14.6, openpyxl 3.1.5. |
| Optional XGBoost import | PASS | XGBoost 3.1.3 imported; XGBoost regression and classification were included in the integration model loops. |
| Optional SHAP import | PASS | SHAP 0.50.0 imported; SHAP is invoked on demand and has a permutation-importance fallback. |
| Unit tests | PASS | 6 unit tests passed. |
| Integration tests | PASS | 3 integration tests passed, including chronological regression, classification and five-fold walk-forward validation on test-only synthetic data. |
| Uploaded workbook smoke test | PASS | All 8 worksheets loaded from the uploaded Excel file; daily/monthly/external tables parsed and feature-engineered. |
| Daily leakage guard | PASS | The daily model matrix excluded all forward target fields; 42 current-session numeric feature columns were available for the 22-row sample. |
| Insufficient-data gate | PASS | Daily supervised modelling returned `insufficient_data` for 21 labelled rows (80 required). |
| SIP research gate | PASS | Econometric routine returned `missing_dependent_variable`; no proxy outcome was substituted. |
| Streamlit application launch | NOT EXECUTED | Streamlit is not installed in the execution environment. Installing it was attempted, but package index access failed due unavailable DNS/network. The app source compiled successfully, but an interactive server/browser smoke test could not be performed here. |

**Automated test result:** 9 passed, 0 failed. The tests use synthetic data only for software verification; those observations are not presented as market data or performance results.

## Exact test commands

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## What the tests cover

- Daily technical feature generation and next-session target shifting.
- Exclusion of forward target columns from the daily predictor matrix.
- Monthly SIP × FII-selling interaction construction and monthly row preservation.
- Duplicate/missing-value audit behavior.
- Chronological date-order guard (overlap rejected).
- Refusal to train a daily model below the minimum sample threshold.
- Refusal to run SIP hypothesis regression when a valid monthly Nifty return outcome is absent.
- Chronological regression model pipeline and baseline metrics.
- Chronological classification model pipeline and classification metrics.
- Expanding-window TimeSeriesSplit walk-forward evaluation.

## Uploaded workbook smoke test

- 8 workbook sheets discovered and parsed.
- Daily Nifty modelling table: 22 rows, 26 columns, 21 non-missing shifted target returns.
- Monthly SIP/FII/DII and derived controls: 130 months, 5 missing SIP inflows.
- External VIX/S&P 500 history: 2,824 dates.
- No duplicate rows or duplicate dates in the three main datasets.
- Daily ML correctly blocked; monthly hypothesis regression correctly blocked for missing dependent variable.

## Outstanding verification

The Streamlit runtime itself must be installed in the target Windows or Streamlit Cloud environment. After installing `requirements.txt`, run `streamlit run app.py` and upload the source workbook. Interactive rendering and browser-side behavior are not claimed as tested in this environment.
