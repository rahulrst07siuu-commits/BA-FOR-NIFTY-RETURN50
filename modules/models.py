import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

REGRESSORS={'Linear Regression':LinearRegression(),'Ridge':Ridge(alpha=1.0),'Lasso':Lasso(alpha=.01,max_iter=10000),'Random Forest':RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42),'Extra Trees':ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=42),'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,l2_regularization=1.0,random_state=42)}
CLASSIFIERS={'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced'),'Random Forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=42),'Extra Trees':ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced',random_state=42),'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=100,l2_regularization=1.0,random_state=42)}
try:
    from xgboost import XGBRegressor, XGBClassifier
    REGRESSORS['XGBoost']=XGBRegressor(n_estimators=150,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,reg_lambda=2,objective='reg:squarederror',n_jobs=2,random_state=42)
    CLASSIFIERS['XGBoost']=XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.03,subsample=.85,colsample_bytree=.85,reg_lambda=2,eval_metric='logloss',n_jobs=2,random_state=42)
except Exception:
    pass
def _pipe(model, scale=False):
    steps=[('imputer',SimpleImputer(strategy='median'))]
    if scale: steps.append(('scaler',StandardScaler()))
    steps.append(('model',model)); return Pipeline(steps)
def chronological_regression(X,y,min_rows=80):
    data=pd.concat([X,y.rename('target')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(subset=['target'])
    if len(data)<min_rows: return {'status':'insufficient_data','rows':len(data),'required':min_rows,'results':pd.DataFrame(),'models':{},'predictions':None}
    data=data.sort_index(); X=data.drop(columns='target'); y=data.target
    split=max(int(len(data)*.8),len(data)-max(20,int(len(data)*.2)))
    split=min(split,len(data)-10); Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
    n_splits=min(5,max(2,len(Xtr)//20)); cv=TimeSeriesSplit(n_splits=n_splits)
    rows=[]; fitted={}; preds={}
    for name,est in REGRESSORS.items():
        pipe=_pipe(est,name in ['Ridge','Lasso'])
        try:
            cv_scores=-cross_val_score(pipe,Xtr,ytr,cv=cv,scoring='neg_root_mean_squared_error')
            pipe.fit(Xtr,ytr); p=pipe.predict(Xte); fitted[name]=pipe; preds[name]=p
            r2=r2_score(yte,p); adj=1-(1-r2)*(len(yte)-1)/(len(yte)-X.shape[1]-1) if len(yte)>X.shape[1]+1 else np.nan
            rows.append({'Model':name,'R2':r2,'Adjusted_R2':adj,'MAE':mean_absolute_error(yte,p),'MSE':mean_squared_error(yte,p),'RMSE':mean_squared_error(yte,p)**.5,'CV_RMSE_Mean':cv_scores.mean(),'CV_RMSE_SD':cv_scores.std()})
        except Exception as e: rows.append({'Model':name,'Error':str(e)})
    baseline=np.repeat(ytr.mean(),len(yte)); base_r2=r2_score(yte,baseline); base_adj=1-(1-base_r2)*(len(yte)-1)/(len(yte)-X.shape[1]-1) if len(yte)>X.shape[1]+1 else np.nan
    rows.append({'Model':'Training-mean baseline','R2':base_r2,'Adjusted_R2':base_adj,'MAE':mean_absolute_error(yte,baseline),'MSE':mean_squared_error(yte,baseline),'RMSE':mean_squared_error(yte,baseline)**.5,'CV_RMSE_Mean':np.nan,'CV_RMSE_SD':np.nan})
    return {'status':'ok','rows':len(data),'train_rows':len(Xtr),'test_rows':len(Xte),'results':pd.DataFrame(rows).sort_values('RMSE',na_position='last'),'models':fitted,'predictions':preds,'X_test':Xte,'y_test':yte,'split_index':split}
def chronological_classification(X,y,min_rows=80):
    data=pd.concat([X,y.rename('target')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(subset=['target']); data=data.sort_index()
    if len(data)<min_rows or data.target.nunique()<2: return {'status':'insufficient_data','rows':len(data),'required':min_rows,'results':pd.DataFrame(),'models':{}}
    split=min(max(int(len(data)*.8),len(data)-20),len(data)-10); Xtr,Xte=data.drop(columns='target').iloc[:split],data.drop(columns='target').iloc[split:]; ytr,yte=data.target.iloc[:split],data.target.iloc[split:]
    rows=[]; models={}
    for name,est in CLASSIFIERS.items():
        pipe=_pipe(est,name=='Logistic Regression')
        try:
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1] if hasattr(pipe,'predict_proba') and len(pipe.classes_)==2 else None
            row={'Model':name,'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0),'ROC_AUC':roc_auc_score(yte,proba) if proba is not None and yte.nunique()==2 else np.nan}
            rows.append(row); models[name]=pipe
        except Exception as e: rows.append({'Model':name,'Error':str(e)})
    return {'status':'ok','results':pd.DataFrame(rows),'models':models,'X_test':Xte,'y_test':yte}
