import numpy as np
import pandas as pd

def _rsi(s, period=14):
    delta=s.diff(); gain=delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean(); loss=-delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); return 100-(100/(1+rs))

def build_daily_features(df):
    d=df.copy(); d.columns=[str(c).strip() for c in d.columns]
    if 'Date' in d: d['Date']=pd.to_datetime(d['Date'],errors='coerce'); d=d.sort_values('Date')
    for c in ['Nifty_Open','Nifty_High','Nifty_Low','Nifty_Close','Volume_Shares_Approx','IndiaVIX_Close','SP500_Return_Calc_Pct']:
        if c in d: d[c]=pd.to_numeric(d[c],errors='coerce')
    close=d['Nifty_Close'] if 'Nifty_Close' in d else pd.Series(dtype=float)
    d['return_pct']=close.pct_change()*100
    for n in [1,2,5]: d[f'return_lag_{n}']=d['return_pct'].shift(n)
    for n in [5,10,20,50]: d[f'sma_{n}']=close.rolling(n,min_periods=n).mean()
    for n in [9,20]: d[f'ema_{n}']=close.ewm(span=n,adjust=False,min_periods=n).mean()
    d['rsi_14']=_rsi(close)
    ema12=close.ewm(span=12,adjust=False,min_periods=12).mean(); ema26=close.ewm(span=26,adjust=False,min_periods=26).mean()
    d['macd']=ema12-ema26; d['macd_signal']=d['macd'].ewm(span=9,adjust=False,min_periods=9).mean()
    mid=close.rolling(20,min_periods=20).mean(); sd=close.rolling(20,min_periods=20).std()
    d['bb_upper']=mid+2*sd; d['bb_lower']=mid-2*sd; d['bb_width_pct']=(d['bb_upper']-d['bb_lower'])/mid*100
    prev=close.shift(1); tr=pd.concat([(d['Nifty_High']-d['Nifty_Low']).abs(),(d['Nifty_High']-prev).abs(),(d['Nifty_Low']-prev).abs()],axis=1).max(axis=1)
    d['atr_14']=tr.rolling(14,min_periods=14).mean()
    d['momentum_10_pct']=close.pct_change(10)*100
    d['realized_vol_5d_pct']=d['return_pct'].rolling(5,min_periods=5).std()
    d['realized_vol_20d_pct']=d['return_pct'].rolling(20,min_periods=20).std()
    if 'IndiaVIX_Close' in d: d['india_vix']=d['IndiaVIX_Close']
    if 'SP500_Return_Calc_Pct' in d: d['sp500_prior_session_return']=d['SP500_Return_Calc_Pct']
    d['target_next_return_pct']=d['return_pct'].shift(-1)
    d['target_next_close']=close.shift(-1)
    return d

def build_monthly_research(monthly, derived=None):
    m=monthly.copy(); m['Month']=pd.to_datetime(m['Month'],errors='coerce'); m=m.sort_values('Month')
    if derived is not None and 'Month' in derived:
        x=derived.copy(); x['Month']=pd.to_datetime(x['Month'],errors='coerce')
        extra=[c for c in x.columns if c!='Month' and c not in m.columns]
        m=m.merge(x[['Month']+extra],on='Month',how='left',validate='one_to_one')
    m['FII Selling (0/1)']=pd.to_numeric(m['FII Selling (0/1)'],errors='coerce')
    if 'SIP Inflow (cr)' in m:
        m['SIP x FII Selling']=pd.to_numeric(m['SIP Inflow (cr)'],errors='coerce')*m['FII Selling (0/1)']
    return m

def daily_model_matrix(feature_frame):
    """Return current-session numeric predictors and next-session return target, excluding all forward labels."""
    target='target_next_return_pct'
    excluded={target,'target_next_close','Target_Next_Session_Return_Pct','Target_Next_Session_Close'}
    X=feature_frame.select_dtypes(include='number').drop(columns=[c for c in excluded if c in feature_frame.columns],errors='ignore')
    y=feature_frame[target].copy() if target in feature_frame else pd.Series(index=feature_frame.index,dtype=float,name=target)
    return X.replace([np.inf,-np.inf],np.nan),y
