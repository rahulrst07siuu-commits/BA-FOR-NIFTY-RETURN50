import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_table(model,X,y,scoring='neg_root_mean_squared_error'):
    result=permutation_importance(model,X,y,n_repeats=10,random_state=42,scoring=scoring)
    return pd.DataFrame({'feature':X.columns,'importance_mean':result.importances_mean,'importance_std':result.importances_std}).sort_values('importance_mean',ascending=False)

def coefficients_table(pipeline, feature_names):
    model=pipeline.named_steps['model']
    if not hasattr(model,'coef_'): return pd.DataFrame()
    coef=model.coef_.ravel()
    return pd.DataFrame({'feature':feature_names,'coefficient':coef}).assign(abs_coefficient=lambda x:x.coefficient.abs()).sort_values('abs_coefficient',ascending=False)


def shap_values_optional(pipeline, X, max_rows=100):
    """Return SHAP values when SHAP imports and the estimator is supported."""
    try:
        import shap
        model=pipeline.named_steps['model']
        transformed=pipeline[:-1].transform(X.iloc[:max_rows])
        if hasattr(model,'feature_importances_'):
            explainer=shap.TreeExplainer(model); values=explainer.shap_values(transformed)
        else:
            explainer=shap.LinearExplainer(model,transformed); values=explainer.shap_values(transformed)
        return {'status':'ok','values':values,'base_values':getattr(explainer,'expected_value',None)}
    except Exception as exc:
        return {'status':'unavailable','message':str(exc)}
