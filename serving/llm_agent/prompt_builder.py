def build_prompt(metrics, drift, canary_data, model_version, shap_reasons):
    return f"""
You are an ML Governance Analyst writing for business stakeholders.

IMPORTANT RULES:
- If latency metrics are missing (None), DO NOT reject the deployment.
- In absence of traffic, mark the decision as DEFERRED due to no traffic.
- Do NOT hallucinate latency failures.

Model Version: {model_version}

Evaluation Metrics:
ROC-AUC: {metrics.get("roc_auc")}
F1 Score: {metrics.get("f1")}
Accuracy: {metrics.get("accuracy")}

Data Drift (PSI):
{drift}

Latency Evaluation:
Stable P95: {canary_data.get("stable_p95")}
Canary P95: {canary_data.get("canary_p95")}
Decision: {canary_data.get("decision")}

Top Churn Drivers (business explanations):
{shap_reasons}

Generate STRICT JSON with keys:
- Retraining Reason
- Deployment Safety Summary
- Risk Assessment
- Top Churn Drivers
- Executive Summary

Respond ONLY in valid JSON.
"""