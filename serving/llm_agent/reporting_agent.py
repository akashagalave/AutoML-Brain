import os
import json
import requests
from openai import OpenAI
from prompt_builder import build_prompt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

REPORT_DIR = "/app/reports/governance"
PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def query_p95(track):
    pod_regex = "inference-canary.*" if track == "canary" else "inference-stable.*"

    query = f"""
    histogram_quantile(
      0.95,
      sum(
        rate(
          churn_request_latency_seconds_bucket{{path="/predict",pod=~"{pod_regex}"}}[5m]
        )
      ) by (le)
    )
    """

    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )

    data = response.json()
    if data.get("status") != "success" or not data["data"]["result"]:
        return None

    return float(data["data"]["result"][0]["value"][1])


def generate_pdf(report):
    os.makedirs(REPORT_DIR, exist_ok=True)

    file_path = f"{REPORT_DIR}/deployment_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("AutoML Brain – Deployment Governance Report", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 0.4 * inch))

    def section(title_text):
        elements.append(Paragraph(f"<b>{title_text}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))

    def bullet(text):
        elements.append(Paragraph(f"- {text}", styles["Normal"]))

    # 1️⃣ Retraining Reason
    section("Retraining Reason")
    bullet(report["Retraining Reason"])

    # 2️⃣ Deployment Safety Summary
    section("Deployment Safety Summary")
    safety = report["Deployment Safety Summary"]

    bullet(f"Model Version: {safety['model_version']}")
    metrics = safety["evaluation_metrics"]
    bullet(f"ROC-AUC: {metrics['ROC_AUC']:.3f}")
    bullet(f"F1 Score: {metrics['F1_Score']:.3f}")
    bullet(f"Accuracy: {metrics['Accuracy']:.3f}")

    canary = safety["canary_evaluation"]
    bullet(f"Canary Decision: {canary['decision']}")

    # 3️⃣ Risk Assessment
    section("Risk Assessment")
    risk = report["Risk Assessment"]

    bullet(f"Data Drift Level: {risk['data_drift']}")
    bullet(f"Model Recommendation: {risk['model_performance']['recommendation']}")
    bullet(f"Canary Status: {risk['canary_release']}")

    # 4️⃣ Executive Summary
    section("Executive Summary")
    exec_sum = report["Executive Summary"]

    bullet(exec_sum["overall_status"])
    elements.append(Spacer(1, 0.1 * inch))
    bullet("Next Steps:")
    for step in exec_sum["next_steps"]:
        bullet(f"• {step}")

    doc.build(elements)


def main():
    metrics = load_json("/app/reports/evaluation/metrics.json")
    drift = load_json("/app/reports/drift/psi_report.json")

    stable_p95 = query_p95("stable")
    canary_p95 = query_p95("canary")

    decision = (
        "PROMOTED"
        if stable_p95 and canary_p95 and canary_p95 <= stable_p95
        else "REJECTED"
    )

    canary_data = {
        "stable_p95": stable_p95,
        "canary_p95": canary_p95,
        "decision": decision,
    }

    prompt = build_prompt(metrics, drift, canary_data, "Latest Production")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a professional ML governance analyst. Respond ONLY in valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("LLM returned empty response")

    report_json = json.loads(raw)

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(f"{REPORT_DIR}/deployment_report.json", "w") as f:
        json.dump(report_json, f, indent=4)

    generate_pdf(report_json)

    print("✅ Deployment report generated successfully")


if __name__ == "__main__":
    main()