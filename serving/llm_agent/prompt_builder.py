def build_prompt(metrics, drift, canary_data, model_version):
    return f"""
You are an AI ML Governance Analyst.

Model Version: {model_version}

Evaluation Metrics:
ROC-AUC: {metrics.get("roc_auc")}
F1 Score: {metrics.get("f1")}
Accuracy: {metrics.get("accuracy")}

Drift Metrics (PSI):
{drift}

Canary Evaluation:
Stable P95 Latency: {canary_data.get("stable_p95")}
Canary P95 Latency: {canary_data.get("canary_p95")}
Decision: {canary_data.get("decision")}

Generate a JSON object with:
- retraining_reason
- deployment_safety_summary
- risk_assessment
- executive_summary
"""