import numpy as np, pandas as pd

def sip_regression(monthly, dependent=None):
    """OLS interaction model. Requires a genuine contemporaneous monthly Nifty return outcome column."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as e: return {'status':'dependency_missing','message':str(e)}
    candidates=[dependent] if dependent else ['Nifty Monthly Return (%)','Monthly Nifty Return (%)','Nifty_Return_Monthly_Pct']
    ycol=next((c for c in candidates if c and c in monthly.columns),None)
    if not ycol: return {'status':'missing_dependent_variable','message':'No monthly Nifty return outcome is present. Previous Month Nifty Return (%) is a lagged control, not a substitute for the current-month dependent variable. Upload a monthly Nifty return series to test H0/H1.'}
    d=monthly.copy(); d['SIP Inflow (cr)']=pd.to_numeric(d['SIP Inflow (cr)'],errors='coerce'); d['FII Net Flow (cr)']=pd.to_numeric(d['FII Net Flow (cr)'],errors='coerce'); d['FII Selling (0/1)']=pd.to_numeric(d['FII Selling (0/1)'],errors='coerce')
    d['SIP_x_FII_Selling']=d['SIP Inflow (cr)']*d['FII Selling (0/1)']
    controls=[c for c in ['DII Net Flow (cr)','Previous Month Nifty Return (%)','Previous 3-Month Return Volatility (%)'] if c in d]
    cols=[ycol,'SIP Inflow (cr)','FII Net Flow (cr)','FII Selling (0/1)','SIP_x_FII_Selling']+controls
    d=d[cols].apply(pd.to_numeric,errors='coerce').dropna()
    if len(d)<36: return {'status':'insufficient_data','rows':len(d),'required':36}
    X=sm.add_constant(d.drop(columns=ycol)); y=d[ycol]
    fit=sm.OLS(y,X).fit(cov_type='HC3'); bp=het_breuschpagan(fit.resid,fit.model.exog)
    vif=[]
    for i,c in enumerate(X.columns):
        if c=='const': continue
        try: v=variance_inflation_factor(X.values,i)
        except Exception: v=np.nan
        vif.append({'variable':c,'VIF':float(v)})
    return {'status':'ok','model':fit,'summary':fit.summary().as_text(),'coefficients':fit.summary2().tables[1],'vif':pd.DataFrame(vif),'breusch_pagan_p':float(bp[1]),'n':len(d),'dependent':ycol,'adf':adfuller(y.dropna())[1] if len(y)>20 else np.nan,'kpss':kpss(y.dropna(),regression='c',nlags='auto')[1] if len(y)>20 else np.nan}
