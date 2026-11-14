import numpy as np

def calibrated_interval(point_forecast, residuals, alpha=.10):
    r=np.asarray(residuals,dtype=float); r=r[np.isfinite(r)]
    if len(r)<10: return None
    q=np.quantile(np.abs(r),1-alpha); return (float(point_forecast-q),float(point_forecast+q),float(q))

def classify_risk(vix=None, realized_vol=None, interval_width=None):
    reasons=[]; score=0
    if vix is not None and np.isfinite(vix):
        if vix>=25: score+=2; reasons.append(f'India VIX {vix:.2f} is elevated (≥25)')
        elif vix>=18: score+=1; reasons.append(f'India VIX {vix:.2f} is above the calm threshold (≥18)')
        else: reasons.append(f'India VIX {vix:.2f} is below 18')
    if realized_vol is not None and np.isfinite(realized_vol):
        if realized_vol>=1.5: score+=2; reasons.append(f'Realised 5-day volatility {realized_vol:.2f}% is elevated')
        elif realized_vol>=.8: score+=1; reasons.append(f'Realised 5-day volatility {realized_vol:.2f}% is moderate')
    if interval_width is not None and np.isfinite(interval_width) and interval_width>=2: score+=1; reasons.append('Forecast interval is wide')
    label='High Risk' if score>=3 else 'Cautious' if score==2 else 'Neutral / Wait and Watch' if score==1 else 'Positive / Lower Risk'
    return {'scenario':label,'risk_score_rule_based_not_probability':score,'reasons':reasons}
