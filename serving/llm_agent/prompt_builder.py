def build_prompt(metrics, drift, canary_data, model_version, shap_reasons):
    """
    Builds the LLM governance report prompt.

    Fix from original:
    - Added pr_auc to evaluation metrics (most important metric for imbalanced
      churn classification — ROC-AUC alone misses precision-recall tradeoff)
    - Added best_threshold to show the operating point used
    - Moved JSON instruction to system message only (no duplicate in user prompt)
    - Cleaner section formatting for better LLM parsing
    """

    if isinstance(drift, dict):
        feature_scores = drift.get("feature_scores", {})
        if feature_scores:
            top_drift = sorted(feature_scores.items(), key=lambda x: -x[1])[:5]
            drift_summary = "\n".join(
                f"  {feat}: PSI={score:.4f}" for feat, score in top_drift
            )
            drift_summary += f"\n  max_psi={drift.get('max_psi', 'N/A')}"
            drift_summary += f"\n  drift_detected={drift.get('drift_detected', 'N/A')}"
        else:
            drift_summary = str(drift)
    else:
        drift_summary = str(drift)

    return f"""
You are an ML Governance Analyst writing deployment reports for business stakeholders.
Your audience includes both technical leads and non-technical executives.

STRICT RULES:
- If latency metrics are missing (None or null), mark decision as DEFERRED, never REJECTED.
- Do NOT hallucinate latency values or failure reasons.
- Base Risk Assessment only on the data provided below.
- Use plain business language in the Executive Summary.

=== MODEL DETAILS ===
Model Version: {model_version}
Problem Domain: Customer Churn Prediction (binary classification)

=== EVALUATION METRICS ===
ROC-AUC:          {metrics.get("roc_auc", "N/A")}
PR-AUC:           {metrics.get("pr_auc", "N/A")}
F1 Score:         {metrics.get("f1", "N/A")}
Accuracy:         {metrics.get("accuracy", "N/A")}
Best Threshold:   {metrics.get("best_threshold", "N/A")}

Note: PR-AUC is the primary metric for this imbalanced dataset.
PR-AUC > 0.80 is considered production-ready for this use case.

=== DATA DRIFT (PSI) ===
{drift_summary}

PSI Interpretation: < 0.10 = stable | 0.10-0.20 = moderate | > 0.20 = significant drift

=== CANARY DEPLOYMENT METRICS ===
Stable P95 Latency: {canary_data.get("stable_p95")} seconds
Canary P95 Latency: {canary_data.get("canary_p95")} seconds
Deployment Decision: {canary_data.get("decision")}

=== TOP CHURN DRIVERS (SHAP) ===
{shap_reasons}

=== OUTPUT FORMAT ===
Return STRICT JSON with exactly these keys:
{{
  "Retraining Reason": "string — why retraining was triggered",
  "Deployment Safety Summary": "string — is this deployment safe to promote?",
  "Risk Assessment": "string — low/medium/high with explanation",
  "Top Churn Drivers": [
    {{"Driver": "feature_name", "Explanation": "business explanation"}}
  ],
  "Executive Summary": "2-3 sentence plain English summary for non-technical stakeholders"
}}
"""
