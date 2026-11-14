# Source workbook data audit

Source: `Nifty50_Daily_and_Monthly_Dataset_V2(1).xlsx` (inspected in-place; not copied into the project ZIP).

## Workbook inventory

- **Read Me** — 13 data rows × 2 columns parsed.
- **Daily Nifty Model Data** — 22 data rows × 26 columns parsed.
- **Monthly SIP FII DII** — 130 data rows × 14 columns parsed.
- **Monthly Derived Vars** — 130 data rows × 4 columns parsed.
- **Full Daily VIX SP500** — 2,824 data rows × 13 columns parsed.
- **Source Audit** — 4 data rows × 7 columns parsed.
- **Variable Register** — 13 data rows × 6 columns parsed.
- **Model Plan** — 6 data rows × 3 columns parsed.

## Dataset audit

### Daily Nifty model data
- Rows / columns: 22 / 26
- Coverage: 2026-10-14 to 2026-11-13
- Frequency: daily/irregular
- Duplicate rows / dates: 0 / 0
- Missing cells: 35
- Missing by field:
  - `Nifty_Return_Calc_Pct`: 1
  - `Nifty_Gap_Pct`: 1
  - `Nifty_Return_Lag1_Pct`: 2
  - `Nifty_Return_Lag2_Pct`: 3
  - `Nifty_Return_Lag5_Pct`: 6
  - `Nifty_Rolling_Volatility_5d_Pct`: 5
  - `Nifty_SMA_5`: 4
  - `Nifty_SMA_10`: 9
  - `IndiaVIX_Return_Calc_Pct`: 1
  - `Target_Next_Session_Return_Pct`: 1
  - `Target_Next_Session_Close`: 1
  - `Target_Next_Session_Date`: 1
- IQR screening counts (descriptive flags, not confirmed errors):
  - `Volume_Shares_Approx`: 2
  - `Nifty_Range_Pct`: 1
  - `Nifty_Gap_Pct`: 1
  - `SP500_Return_Calc_Pct`: 3
  - `SP500_Sessions_Lagged_By_Calendar_Days`: 5
### Monthly SIP FII DII + derived controls
- Rows / columns: 130 / 17
- Coverage: 2016-01-01 to 2026-10-01
- Frequency: monthly
- Duplicate rows / dates: 0 / 0
- Missing cells: 21
- Missing by field:
  - `SIP Inflow (cr)`: 5
  - `FII Selling x SIP (cr)`: 5
  - `Previous Month Nifty Return (%)`: 2
  - `Previous 3-Month Return Volatility (%)`: 4
  - `SIP x FII Selling`: 5
- IQR screening counts (descriptive flags, not confirmed errors):
  - `FII Net Flow (cr)`: 7
  - `DII Gross Sales (cr)`: 1
  - `DII Net Flow (cr)`: 7
  - `FII_Net_Check_cr`: 7
  - `DII_Net_Check_cr`: 7
  - `FII_Net_Reconciliation_Diff_cr`: 9
  - `DII_Net_Reconciliation_Diff_cr`: 10
  - `Previous Month Nifty Return (%)`: 3
  - `Previous 3-Month Return Volatility (%)`: 7
### Full Daily VIX SP500
- Rows / columns: 2824 / 13
- Coverage: 2016-01-01 to 2026-11-13
- Frequency: daily/irregular
- Duplicate rows / dates: 0 / 0
- Missing cells: 1394
- Missing by field:
  - `SP500_Close`: 99
  - `SP500_Open`: 99
  - `SP500_High`: 99
  - `SP500_Low`: 99
  - `SP500_Change_Source_Pct`: 99
  - `SP500_Return_Calc_Pct`: 100
  - `IndiaVIX_Close`: 133
  - `IndiaVIX_Open`: 133
  - `IndiaVIX_High`: 133
  - `IndiaVIX_Low`: 133
  - `IndiaVIX_Change_Source_Pct`: 133
  - `IndiaVIX_Return_Calc_Pct`: 134
- IQR screening counts (descriptive flags, not confirmed errors):
  - `SP500_Change_Source_Pct`: 181
  - `SP500_Return_Calc_Pct`: 182
  - `IndiaVIX_Close`: 105
  - `IndiaVIX_Open`: 105
  - `IndiaVIX_High`: 108
  - `IndiaVIX_Low`: 102
  - `IndiaVIX_Change_Source_Pct`: 128
  - `IndiaVIX_Return_Calc_Pct`: 123

## Findings and modelling implications
- The daily modelling sheet has **22 trading sessions** (14-Oct-2026 to 13-Nov-2026) and 21 non-missing shifted next-session targets. This is insufficient for robust ML, five-fold time-series validation, calibration or a meaningful untouched holdout. The application blocks supervised daily training below 80 labelled rows.
- Daily OHLC, VIX and S&P 500 numeric columns were parsed as numeric; `Volume_Raw` is retained as a raw source string while `Volume_Shares_Approx` is numeric. No missing values were replaced with zero.
- Monthly SIP/FII/DII has 130 observations (Jan-2016 to Oct-2026). SIP inflows are missing in Jan-2016, Feb-2016, Mar-2016, Sep-2026 and Oct-2026 (5 rows). Companion derived controls have 2 missing prior-month return observations and 4 missing 3-month volatility observations.
- Monthly `FII Net Flow (cr)` and `FII Selling (0/1)` are present. However, no current-month Nifty return dependent variable is included. `Previous Month Nifty Return (%)` is a lagged control and is not substituted as the outcome. Consequently the SIP Shield H0/H1 interaction regression is correctly disabled pending a genuine monthly Nifty return series.
- Full VIX/S&P 500 external history has 2,824 dates (2016-01-01 to 2026-11-13). S&P 500 close has 99 missing dates and calculated return has 100; India VIX close has 133 missing and calculated return has 134. This long external table does not expand the 22-row daily Nifty target dataset.
- No duplicate rows or duplicate dates were found in the three main datasets. IQR flags are descriptive and require source-level review; they are not automatically removed or winsorized.
- Daily and monthly tables remain independent. Monthly values are never repeated across daily observations.

## Data readiness

| Workstream | Status | Reason |
|---|---|---|
| Daily feature engineering | Available | OHLCV/VIX/S&P inputs present for 22 sessions; indicators with longer lookbacks have warm-up missingness |
| Daily supervised regression/classification | Blocked | Only 21 labelled target observations; below 80-row safety gate |
| Walk-forward / final holdout evaluation | Blocked | Sample cannot support stable folds and untouched test |
| Forecast intervals / probabilities | Not issued | No validated fitted model or calibration residuals |
| Monthly SIP Shield regression | Blocked | Missing current-month Nifty return dependent variable |
| Monthly descriptive audit | Available | 130 monthly rows; SIP has 5 missing values |

## Unit and timing notes

- Monetary flow fields use the workbook's `cr` suffix (crore).
- Daily return fields are percent fields based on source column names. Validate source calculation definitions before external comparison.
- The S&P 500 reference date is intended as the prior completed US session; verify that every reference date is strictly before its Nifty session in any expanded future dataset.
- The application preserves source observations and exposes missingness. Imputation is only applied within training pipelines, and only after the minimum training gate is passed.