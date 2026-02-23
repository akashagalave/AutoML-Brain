import os
import json
import csv
import requests
from openai import OpenAI
from prompt_builder import build_prompt
from shap_explanations import SHAP_EXPLANATIONS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

REPORT_DIR = "/app/reports/governance"
SHAP_CSV = "/app/reports/shap/shap_global_importance.csv"
PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_top_shap_reasons(csv_path, top_k=5):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = sorted(
            reader,
            key=lambda x: float(x["mean_abs_shap"]),
            reverse=True,
        )[:top_k]

    reasons = []
    for row in rows:
        feature = row["feature"]
        explanation = SHAP_EXPLANATIONS.get(
            feature,
            "This feature has a statistically significant impact on churn."
        )
        reasons.append({
            "Driver": feature,
            "Explanation": explanation
        })

    return reasons


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

    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )

    data = r.json()
    if data["status"] != "success" or not data["data"]["result"]:
        return None

    return float(data["data"]["result"][0]["value"][1])


def generate_pdf(report):
    os.makedirs(REPORT_DIR, exist_ok=True)

    doc = SimpleDocTemplate(
        f"{REPORT_DIR}/deployment_report.pdf", pagesize=A4
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("AutoML Brain – Deployment Governance Report", styles["Title"]))
    elements.append(Spacer(1, 0.4 * inch))

    def section(title):
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))

    section("Retraining Reason")
    elements.append(Paragraph(report["Retraining Reason"], styles["Normal"]))

    section("Deployment Safety Summary")
    elements.append(Paragraph(str(report["Deployment Safety Summary"]), styles["Normal"]))

    section("Risk Assessment")
    elements.append(Paragraph(str(report["Risk Assessment"]), styles["Normal"]))

    section("Top Reasons for Customer Churn")
    for r in report["Top Churn Drivers"]:
        elements.append(
            Paragraph(f"- <b>{r['Driver']}:</b> {r['Explanation']}", styles["Normal"])
        )

    section("Executive Summary")
    elements.append(Paragraph(report["Executive Summary"], styles["Normal"]))

    doc.build(elements)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    metrics = load_json("/app/reports/evaluation/metrics.json")
    drift = load_json("/app/reports/drift/psi_report.json")

    stable_p95 = query_p95("stable")
    canary_p95 = query_p95("canary")

    if stable_p95 is None or canary_p95 is None:
        decision = "DEFERRED (NO TRAFFIC)"
    elif canary_p95 <= stable_p95:
        decision = "PROMOTED"
    else:
        decision = "REJECTED"

    canary_data = {
        "stable_p95": stable_p95,
        "canary_p95": canary_p95,
        "decision": decision,
    }

    shap_reasons = extract_top_shap_reasons(SHAP_CSV)

    prompt = build_prompt(
        metrics,
        drift,
        canary_data,
        "Latest Production",
        shap_reasons,
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Respond ONLY in valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    report = json.loads(response.choices[0].message.content)

    with open(f"{REPORT_DIR}/deployment_report.json", "w") as f:
        json.dump(report, f, indent=4)

    generate_pdf(report)

    print("✅ Governance report generated successfully")


if __name__ == "__main__":
    main()