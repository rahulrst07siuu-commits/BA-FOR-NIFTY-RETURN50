import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def walk_forward_rmse(estimator, X, y, n_splits=5):
    if len(X)<n_splits+5: return {'status':'insufficient_data','folds':[]}
    tscv=TimeSeriesSplit(n_splits=min(n_splits,max(2,len(X)//5))); folds=[]
    for i,(tr,te) in enumerate(tscv.split(X),1):
        model=estimator
        model.fit(X.iloc[tr],y.iloc[tr]); pred=model.predict(X.iloc[te])
        folds.append({'fold':i,'train_end_index':int(tr[-1]),'test_start_index':int(te[0]),'test_end_index':int(te[-1]),'rmse':float(mean_squared_error(y.iloc[te],pred)**.5)})
    return {'status':'ok','folds':folds,'mean_rmse':float(np.mean([x['rmse'] for x in folds]))}

def assert_chronological(train_dates,test_dates):
    if max(train_dates)>=min(test_dates): raise ValueError('Leakage: training dates overlap or follow test dates')
    return True
