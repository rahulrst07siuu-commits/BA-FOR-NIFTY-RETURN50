def managerial_decision(forecast_available, direction_probability=None, risk=None, model_agreement=None):
    if not forecast_available:
        return {'decision':'Insufficient evidence — do not issue a model-led directional view.','rationale':'The daily history is too short for credible chronological validation.'}
    risk=risk or {}; scenario=risk.get('scenario','Unclassified')
    if scenario=='High Risk': action='High Risk: prioritise capital preservation, scenario planning and monitoring; avoid relying on a single-point forecast.'
    elif scenario=='Cautious': action='Cautious: wait for confirmation and monitor volatility and model disagreement.'
    elif direction_probability is not None and direction_probability>=.60: action='Positive scenario: a directional bias may be discussed, but require independent confirmation; this is not a buy recommendation.'
    else: action='Neutral / Wait and Watch: evidence does not support a strong directional stance.'
    return {'decision':action,'rationale':f"Risk regime: {scenario}. Model agreement: {model_agreement if model_agreement is not None else 'not available'}. Probabilities are model estimates, not guarantees."}
