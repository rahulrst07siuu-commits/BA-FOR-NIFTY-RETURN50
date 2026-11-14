import sys
from pathlib import Path
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from config import *
from modules.data_io import load_workbook, read_tabular_upload, data_audit
from modules.features import build_daily_features, build_monthly_research, daily_model_matrix
from modules.models import chronological_regression, chronological_classification, REGRESSORS, CLASSIFIERS
from modules.validation import walk_forward_rmse
from modules.sip_research import sip_regression
from modules.risk import calibrated_interval, classify_risk
from modules.decision_support import managerial_decision
from modules.explainability import permutation_table, coefficients_table, shap_values_optional

st.set_page_config(page_title=APP_TITLE,page_icon='📈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:#F5F8FA;color:#14283D}.block-container{padding-top:1.3rem;max-width:1500px}
section[data-testid="stSidebar"]{background:#10243A} section[data-testid="stSidebar"] *{color:#F5FAFC!important}
div[data-testid="stMetric"]{background:white;border:1px solid #DCE6EB;padding:14px;border-radius:10px}
h1,h2,h3{color:#10243A} .stButton>button{background:#0F9D9A;color:white;border:0;border-radius:7px}
.small-note{color:#5B6D7A;font-size:0.88rem}
</style>''',unsafe_allow_html=True)
st.markdown(f"# 📈 {APP_TITLE}")
st.caption('Daily forecasting and monthly SIP Shield research are independent analytical workstreams.')

@st.cache_data(show_spinner=False)
def load_bytes(data):
    import io
    return load_workbook(io.BytesIO(data))

with st.sidebar:
    st.markdown('## DATA WORKSPACE')
    up=st.file_uploader('Upload source workbook (.xlsx) or daily CSV',type=['xlsx','xls','csv'])
    monthly_up=st.file_uploader('Optional monthly SIP/FII/DII CSV',type=['csv'])
    derived_up=st.file_uploader('Optional monthly controls CSV',type=['csv'])
    st.markdown('---')
    st.caption('Historical uploads only. No paid API or API key required.')

sheets={}; daily=None; monthly=None; derived=None; external=None
if up:
    try:
        if up.name.lower().endswith(('.xlsx','.xls')):
            sheets=load_bytes(up.getvalue())
            daily=sheets.get('Daily Nifty Model Data'); monthly=sheets.get('Monthly SIP FII DII'); derived=sheets.get('Monthly Derived Vars'); external=sheets.get('Full Daily VIX SP500')
        else: daily=read_tabular_upload(up)
    except Exception as e: st.error(f'Could not read upload: {e}')
if monthly_up:
    try: monthly=read_tabular_upload(monthly_up)
    except Exception as e: st.error(f'Monthly CSV error: {e}')
if derived_up:
    try: derived=read_tabular_upload(derived_up)
    except Exception as e: st.error(f'Derived controls CSV error: {e}')
if daily is not None:
    daily=daily.copy(); daily.columns=[str(c).strip() for c in daily.columns]
    if 'Date' in daily: daily['Date']=pd.to_datetime(daily['Date'],errors='coerce'); daily=daily.sort_values('Date')
    try: feat=build_daily_features(daily)
    except Exception as e: feat=daily.copy(); st.warning(f'Feature engineering limited: {e}')
else: feat=pd.DataFrame()
if monthly is not None:
    try: monthly=build_monthly_research(monthly,derived)
    except Exception as e: st.warning(f'Monthly merge warning: {e}')

pages=['Executive Overview','Data Audit','Exploratory Analysis','Technical Indicators','Regression','Classification','Model Comparison','Walk-Forward Validation','Unsupervised Learning','Market Regimes','Explainability','Next-Session Forecast','Risk Analysis','SIP Shield Research','Managerial Decision Support','Methodology and Disclaimer']
page=st.sidebar.radio('NAVIGATION',pages)

def need_daily():
    if feat.empty: st.info('Upload your Excel workbook or daily Nifty CSV in the sidebar to activate this page.'); return False
    return True
def chart_empty(msg): st.info(msg)
def daily_audit(): return data_audit(daily,'Date','Daily Nifty')
def metric_card(label,value,help=None): st.metric(label,value,help=help)

if page=='Executive Overview':
    st.subheader('Executive Overview')
    if not need_daily(): st.stop()
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Daily observations',f'{len(daily):,}')
    c2.metric('Coverage',f"{pd.to_datetime(daily.Date).min():%d %b %Y} – {pd.to_datetime(daily.Date).max():%d %b %Y}" if 'Date' in daily else 'Not identified')
    c3.metric('Monthly observations',f'{len(monthly):,}' if monthly is not None else 'Not uploaded')
    c4.metric('Daily ML readiness','Insufficient' if len(daily)<MIN_DAILY_TRAIN_ROWS else 'Eligible for evaluation')
    if 'Nifty_Close' in daily:
        fig=px.line(daily,x='Date',y='Nifty_Close',title='Nifty 50 closing level',template='plotly_white'); fig.update_traces(line_color=TEAL); st.plotly_chart(fig,use_container_width=True)
    st.warning('Forecast models are deliberately gated by chronological sample sufficiency. The uploaded daily modelling sheet contains only 22 sessions; no credible out-of-sample model can be trained from this sample.')
    st.markdown('**Workstream separation:** daily Nifty observations power daily features; monthly SIP/FII/DII observations remain in the monthly research module and are never expanded onto daily rows.')
elif page=='Data Audit':
    st.subheader('Data Audit & Variable Dictionary')
    if not need_daily(): st.stop()
    tabs=st.tabs(['Daily Nifty','Monthly SIP/FII/DII','External VIX / S&P 500','Workbook sheets'])
    for tab,df,label,datecol in [(tabs[0],daily,'Daily Nifty','Date'),(tabs[1],monthly,'Monthly SIP/FII/DII','Month'),(tabs[2],external,'External VIX / S&P 500','Date')]:
        with tab:
            if df is None: st.info('This table was not found in the uploaded workbook.'); continue
            a=data_audit(df,datecol,label); cols=st.columns(4); cols[0].metric('Rows',a['rows']); cols[1].metric('Columns',a['columns']); cols[2].metric('Missing cells',a['missing_cells']); cols[3].metric('Duplicate rows',a['duplicate_rows'])
            st.write('Date range:',a['start'],'to',a['end'],'| Frequency:',a['frequency'],'| Duplicate dates:',a['duplicate_dates'])
            st.dataframe(pd.DataFrame({'Missing values':pd.Series(a['missing_by_column'])}),use_container_width=True)
            st.dataframe(df.head(10),use_container_width=True)
    with tabs[3]:
        if sheets: st.dataframe(pd.DataFrame([{'Sheet':k,'Rows':len(v),'Columns':len(v.columns)} for k,v in sheets.items()]),use_container_width=True)
        else: st.info('Upload the source Excel workbook to inspect its worksheet inventory.')
    st.markdown('#### Variable dictionary')
    st.dataframe(pd.DataFrame([{'Variable':c,'Data type':str(daily[c].dtype),'Missing':int(daily[c].isna().sum()),'Meaning / treatment':'Source-provided field; verify units and timing before modelling.'} for c in daily.columns]),use_container_width=True)
    st.download_button('Download daily audit CSV',pd.DataFrame([{'Variable':c,'dtype':str(daily[c].dtype),'missing':int(daily[c].isna().sum())} for c in daily.columns]).to_csv(index=False),'daily_variable_dictionary.csv','text/csv')
elif page=='Exploratory Analysis':
    if not need_daily(): st.stop()
    st.subheader('Exploratory Analysis'); num=daily.select_dtypes('number').columns.tolist(); selected=st.multiselect('Numeric variables',num,default=num[:min(5,len(num))])
    if selected: st.plotly_chart(px.line(daily,x='Date',y=selected,template='plotly_white'),use_container_width=True)
    st.dataframe(daily[selected].describe().T if selected else pd.DataFrame(),use_container_width=True)
elif page=='Technical Indicators':
    if not need_daily(): st.stop()
    st.subheader('Technical Indicators'); cols=[c for c in ['Nifty_Close','sma_5','sma_10','sma_20','sma_50','ema_9','ema_20'] if c in feat]
    if cols: st.plotly_chart(px.line(feat,x='Date',y=cols,template='plotly_white',title='Nifty close and moving averages'),use_container_width=True)
    for c in ['rsi_14','macd','macd_signal','bb_upper','bb_lower','atr_14','momentum_10_pct','realized_vol_5d_pct']:
        if c in feat: st.plotly_chart(px.line(feat,x='Date',y=c,template='plotly_white',title=c),use_container_width=True)
elif page in ['Regression','Model Comparison','Walk-Forward Validation','Explainability','Next-Session Forecast']:
    if not need_daily(): st.stop()
    st.subheader(page)
    # Feature matrix excludes forward labels by construction; predictors use current-session data only.
    X,y=daily_model_matrix(feat); features=list(X.columns)
    if len(feat)<MIN_DAILY_TRAIN_ROWS or y.notna().sum()<MIN_DAILY_TRAIN_ROWS:
        st.warning(f'Model training disabled: {len(feat)} daily rows are available and {y.notna().sum()} labelled next-session targets. At least {MIN_DAILY_TRAIN_ROWS} are required by this conservative gate. No synthetic observations are used.')
        st.write('Available feature fields:',features)
        if page=='Next-Session Forecast': st.info('No validated model forecast is issued. Upload a longer historical daily dataset to unlock model training, calibrated intervals and direction probabilities.')
        elif page=='Walk-Forward Validation': st.info('Walk-forward evaluation is unavailable until sufficient sequential observations exist.')
        elif page=='Explainability': st.info('Feature importance is not reported without a trained, validated model.')
        st.stop()
    if page in ['Regression','Model Comparison']:
        res=chronological_regression(X,y)
        if res['status']=='ok':
            st.dataframe(res['results'],use_container_width=True)
            chosen=st.selectbox('Model prediction view',list(res['predictions']))
            plot=pd.DataFrame({'Date':feat.loc[res['X_test'].index,'Date'],'Actual':res['y_test'],'Predicted':res['predictions'][chosen]})
            st.plotly_chart(px.line(plot,x='Date',y=['Actual','Predicted'],template='plotly_white'),use_container_width=True)
        else: st.warning(res)
    elif page=='Walk-Forward Validation':
        from modules.models import _pipe
        from sklearn.linear_model import Ridge
        r=walk_forward_rmse(_pipe(Ridge(),True),X.dropna(axis=1,how='all'),y)
        st.write(r)
    elif page=='Explainability':
        res=chronological_regression(X,y)
        if res['status']=='ok':
            name=st.selectbox('Trained model',list(res['models']))
            try:
                st.dataframe(permutation_table(res['models'][name],res['X_test'],res['y_test']),use_container_width=True)
                if st.button('Calculate optional SHAP values'):
                    sh=shap_values_optional(res['models'][name],res['X_test'])
                    if sh['status']=='ok': st.write('SHAP values calculated for the first 100 holdout rows.'); st.write(np.asarray(sh['values']).shape)
                    else: st.warning(sh['message'])
            except Exception as e: st.error(str(e))
    else:
        reg=chronological_regression(X,y)
        if reg['status']!='ok': st.warning(reg); st.stop()
        scored=reg['results'].dropna(subset=['RMSE']).query("Model != 'Training-mean baseline'")
        if scored.empty: st.warning('No eligible model completed validation.'); st.stop()
        best_name=scored.iloc[0]['Model']
        from sklearn.base import clone
        best=clone(reg['models'][best_name])
        train_mask=y.notna(); X_all=X.loc[train_mask]; y_all=y.loc[train_mask]
        best.fit(X_all,y_all)
        latest=X.iloc[[-1]]
        point=float(best.predict(latest)[0]); current_close=float(pd.to_numeric(daily['Nifty_Close'],errors='coerce').dropna().iloc[-1])
        forecast_close=current_close*(1+point/100)
        holdout_pred=reg['predictions'][best_name]; residuals=reg['y_test'].to_numpy()-holdout_pred
        interval=calibrated_interval(point,residuals)
        low,high=(interval[0],interval[1]) if interval else (None,None)
        # Independently fit direction classifiers on the available labelled history after evaluation.
        cls_y=(y>0).astype(int); cls=chronological_classification(X,cls_y)
        prob=None; cls_name=None
        if cls['status']=='ok':
            ranked=cls['results'].sort_values('F1',ascending=False); cls_name=ranked.iloc[0]['Model']; clf=cls['models'][cls_name]
            clf.fit(X.loc[train_mask],cls_y.loc[train_mask])
            if hasattr(clf,'predict_proba'): prob=float(clf.predict_proba(latest)[0,1])
        model_votes=[]
        for nm,pipe in reg['models'].items():
            try: model_votes.append(float(pipe.predict(latest)[0]))
            except Exception: pass
        agreement=(sum(v>0 for v in model_votes)/len(model_votes)) if model_votes else None
        vix_vals=pd.to_numeric(daily.get('IndiaVIX_Close',pd.Series(dtype=float)),errors='coerce').dropna(); rv_vals=feat.get('realized_vol_5d_pct',pd.Series(dtype=float)).dropna()
        risk=classify_risk(float(vix_vals.iloc[-1]) if len(vix_vals) else None,float(rv_vals.iloc[-1]) if len(rv_vals) else None,(high-low) if low is not None else None)
        c1,c2,c3,c4=st.columns(4); c1.metric('Expected next-session return',f'{point:+.3f}%'); c2.metric('Implied closing level',f'{forecast_close:,.2f}'); c3.metric('Positive direction probability',f'{prob:.1%}' if prob is not None else 'Unavailable'); c4.metric('Negative direction probability',f'{1-prob:.1%}' if prob is not None else 'Unavailable')
        st.metric('Risk scenario',risk['scenario'])
        if len(vix_vals): st.write(f'Latest India VIX: **{float(vix_vals.iloc[-1]):.2f}**')
        if len(rv_vals:=feat.get('realized_vol_5d_pct',pd.Series(dtype=float)).dropna()): st.write(f'Latest realised 5-session volatility (risk proxy, not a separately trained volatility forecast): **{float(rv_vals.iloc[-1]):.3f}%**')
        st.write(f'Best chronological holdout model: **{best_name}** (holdout RMSE {float(scored.iloc[0]["RMSE"]):.4f} percentage points; holdout R² {float(scored.iloc[0]["R2"]):.4f}).')
        if interval: st.write(f'Approximate 90% residual-calibrated interval for next-session return: **{low:+.3f}% to {high:+.3f}%** (absolute residual quantile {interval[2]:.3f} pp).')
        else: st.warning('Not enough holdout residuals to calibrate a prediction interval.')
        st.write(f'Model sign agreement: {agreement:.0%}' if agreement is not None else 'Model agreement unavailable.')
        st.write('Risk drivers:',risk['reasons']); st.caption('Probabilities and intervals are model estimates based on historical validation residuals, not guarantees. The holdout is used for evaluation and interval calibration; the selected model is refit on all labelled history only after model selection.')
elif page=='Classification':
    if not need_daily(): st.stop()
    st.subheader('Market Direction Classification')
    X,return_target=daily_model_matrix(feat); y=(return_target>0).astype(int)
    res=chronological_classification(X,y)
    if res['status']!='ok': st.warning(f"Classification disabled: {res['rows']} rows; {res['required']} required, or only one target class is present.")
    else:
        st.dataframe(res['results'],use_container_width=True)
        from sklearn.metrics import confusion_matrix
        model_name=st.selectbox('Confusion matrix model',list(res['models']))
        cm=confusion_matrix(res['y_test'],res['models'][model_name].predict(res['X_test']))
        st.plotly_chart(px.imshow(cm,text_auto=True,x=['Predicted 0','Predicted 1'],y=['Actual 0','Actual 1'],title=f'{model_name} confusion matrix',template='plotly_white'),use_container_width=True)
elif page=='Unsupervised Learning':
    if not need_daily(): st.stop()
    st.subheader('Unsupervised Learning')
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    X=feat.select_dtypes('number').replace([np.inf,-np.inf],np.nan).drop(columns=['target_next_return_pct','target_next_close'],errors='ignore')
    if len(X)<10: st.info('At least 10 rows are needed for exploratory PCA / K-means.'); st.stop()
    k=st.slider('Clusters',2, min(6,len(X)-1),2)
    z=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()).fit_transform(X)
    pc=PCA(n_components=2).fit_transform(z); lab=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(z)
    st.plotly_chart(px.scatter(x=pc[:,0],y=pc[:,1],color=lab.astype(str),labels={'x':'PC1','y':'PC2','color':'Cluster'},template='plotly_white'),use_container_width=True)
    st.caption('Exploratory clustering only; 22 observations are too few for stable regime inference.')
elif page=='Market Regimes':
    if not need_daily(): st.stop()
    st.subheader('Market Regimes'); v=feat.get('realized_vol_5d_pct',pd.Series(index=feat.index,dtype=float)); q=v.quantile([.33,.67])
    feat['Exploratory volatility bucket']=pd.cut(v,[-np.inf,q.iloc[0],q.iloc[1],np.inf],labels=['Lower','Middle','Higher'])
    st.dataframe(feat[['Date','Nifty_Close','IndiaVIX_Close','realized_vol_5d_pct','Exploratory volatility bucket']].tail(30),use_container_width=True)
    st.caption('Quantile buckets are descriptive labels, not a fitted or validated predictive regime model.')
elif page=='Risk Analysis':
    if not need_daily(): st.stop()
    st.subheader('Risk Analysis'); vix=pd.to_numeric(daily.get('IndiaVIX_Close',pd.Series(dtype=float)),errors='coerce').dropna(); rv=feat.get('realized_vol_5d_pct',pd.Series(dtype=float)).dropna()
    risk=classify_risk(float(vix.iloc[-1]) if len(vix) else None,float(rv.iloc[-1]) if len(rv) else None,None)
    st.metric('Current descriptive risk bucket',risk['scenario']); st.write('Drivers:',risk['reasons']); st.caption('Rule-based descriptive thresholds, not a calibrated probability or investment signal.')
elif page=='SIP Shield Research':
    st.subheader('Monthly SIP Shield Research')
    if monthly is None: st.info('Upload the Excel workbook or a monthly SIP/FII/DII CSV.'); st.stop()
    st.metric('Monthly rows',len(monthly)); st.write('Monthly data range:',monthly['Month'].min(),'to',monthly['Month'].max())
    st.dataframe(monthly.head(20),use_container_width=True)
    if 'SIP Inflow (cr)' in monthly: st.metric('Missing SIP observations',int(monthly['SIP Inflow (cr)'].isna().sum()))
    result=sip_regression(monthly)
    if result['status']=='ok':
        st.text(result['summary']); st.dataframe(result['coefficients'],use_container_width=True); st.dataframe(result['vif'],use_container_width=True); st.write('Breusch–Pagan p-value:',result['breusch_pagan_p']); st.write('ADF p-value:',result['adf'],'| KPSS p-value:',result['kpss']); st.caption('OLS coefficient table uses HC3 heteroskedasticity-robust standard errors.')
        diag=pd.DataFrame({'Fitted':result['model'].fittedvalues,'Residual':result['model'].resid})
        st.plotly_chart(px.scatter(diag,x='Fitted',y='Residual',title='SIP Shield OLS residuals vs fitted',template='plotly_white'),use_container_width=True)
        st.plotly_chart(px.histogram(diag,x='Residual',title='SIP Shield OLS residual distribution',template='plotly_white'),use_container_width=True)
    else: st.warning(result.get('message',result))
    st.markdown('**Hypothesis interpretation:** the interaction coefficient on SIP × FII-selling tests whether the SIP slope differs in FII-selling months. It does not establish causation. The supplied workbook lacks a current-month Nifty return dependent variable, so H0/H1 cannot currently be tested validly.')
elif page=='Managerial Decision Support':
    st.subheader('Managerial Decision Support')
    if not need_daily(): st.stop()
    X,ret_target=daily_model_matrix(feat)
    if int(ret_target.notna().sum())<MIN_DAILY_TRAIN_ROWS:
        st.warning(f'No validated forecast is available from {int(ret_target.notna().sum())} labelled daily observations. The defensible managerial decision is to avoid model-led directional positioning until adequate history is supplied.')
        st.write(managerial_decision(False))
    else:
        reg=chronological_regression(X,ret_target)
        if reg['status']!='ok':
            st.warning('No eligible regression model completed chronological validation.'); st.write(managerial_decision(False))
        else:
            scored=reg['results'].dropna(subset=['RMSE']).query("Model != 'Training-mean baseline'")
            if scored.empty:
                st.warning('No eligible model is available for decision support.'); st.write(managerial_decision(False))
            else:
                best=scored.iloc[0]; vix=pd.to_numeric(daily.get('IndiaVIX_Close',pd.Series(dtype=float)),errors='coerce').dropna(); rv=feat.get('realized_vol_5d_pct',pd.Series(dtype=float)).dropna()
                risk=classify_risk(float(vix.iloc[-1]) if len(vix) else None,float(rv.iloc[-1]) if len(rv) else None)
                cls=chronological_classification(X,(ret_target>0).astype(int)); prob=None
                if cls['status']=='ok':
                    cn=cls['results'].sort_values('F1',ascending=False).iloc[0]['Model']; clf=cls['models'][cn]
                    mask=ret_target.notna(); clf.fit(X.loc[mask],(ret_target.loc[mask]>0).astype(int))
                    if hasattr(clf,'predict_proba'): prob=float(clf.predict_proba(X.iloc[[-1]])[0,1])
                st.write(managerial_decision(True,prob,risk,None))
                st.write(f"Selected model: {best['Model']} | Holdout RMSE: {best['RMSE']:.4f} pp | Holdout R²: {best['R2']:.4f}")
                st.write('Risk drivers:',risk['reasons'])
    st.markdown('**Suggested analytical actions**')
    st.markdown('- Extend the daily Nifty history to several hundred sessions, ideally multiple market regimes.\n- Supply the monthly Nifty return series aligned to the SIP/FII/DII months.\n- Verify SIP coverage, source definitions, units and release timing before econometric estimation.')
else:
    st.subheader('Methodology and Disclaimer')
    st.markdown('''- Chronological train/test splits only; no random shuffling.
- Imputation and scaling belong inside Scikit-learn pipelines, fitted on training folds.
- Expanding-window validation uses TimeSeriesSplit where the sample supports it.
- The final chronological holdout is not used for fitting or feature selection.
- Baselines are mandatory; negative out-of-sample R² is reported as-is.
- Daily forecasting and monthly SIP/FII/DII analysis remain separate frequency domains.
- A prior completed S&P 500 session is acceptable only when its reference date is strictly before the Nifty session; verify this before treating it as live-available.

**Disclaimer:** Educational and research software, not investment advice. Forecasts are uncertain; historical relationships may not persist. No direct buy/sell recommendations or guaranteed returns.''')

st.sidebar.markdown('---'); st.sidebar.caption('Research-grade workflow • No guaranteed forecasts • Upload your own verified data')
