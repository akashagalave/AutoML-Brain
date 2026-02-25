

# 🧠 AutoML Brain

### Autonomous Churn Prediction & ML Governance Platform

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EKS-232F3E?style=for-the-badge\&logo=amazon-aws\&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5?style=for-the-badge\&logo=kubernetes\&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?style=for-the-badge)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge\&logo=prometheus\&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Governance-black?style=for-the-badge)

---

## 📖 Overview

**AutoML Brain** is a **production-grade, autonomous ML platform** built to solve the *real* problems of deploying machine learning systems at scale.

This is **not just a churn prediction model**.

It is an **end-to-end ML lifecycle, deployment, monitoring, and governance system** designed for **subscription-based businesses**, including:

* Streaming platforms (Netflix-like)
* SaaS products
* Telecom subscriptions
* Digital marketplaces

The platform continuously **monitors, validates, deploys, governs, and explains itself** — with minimal human intervention.

---

## ❌ Why Traditional ML Systems Fail

Most ML systems break in production because they:

* Train once and never retrain
* Lack drift detection
* Deploy models unsafely
* Ignore latency and SLA violations
* Depend on manual approvals
* Provide no executive-level visibility

In subscription businesses, **poor ML decisions directly translate to customer churn and revenue loss**.

---

## ✅ What AutoML Brain Solves

AutoML Brain introduces **production-first ML design**, combining:

* Reproducible pipelines (DVC)
* Model lifecycle management (MLflow)
* Kubernetes-native model serving
* Canary deployments with rollback
* Latency & SLA monitoring
* Rule-based ML governance agents
* LLM-generated executive reports

**Outcome:**
➡️ A **self-governing ML system** that is safe, observable, and scalable.

---

## 🧱 High-Level Architecture

```mermaid
flowchart TD
    Client[User / Product Service]

    subgraph Inference
        API[FastAPI Inference API]
        Model[LightGBM Model]
    end

    subgraph ML_Platform
        DVC[DVC Pipeline]
        Train[Training Job]
        MLflow[MLflow Registry]
    end

    subgraph Monitoring
        Prom[Prometheus]
    end

    subgraph Agents
        Deploy[Deployment Agent]
        Gov[Governance Agent]
        LLM[LLM Reporting Agent]
    end

    Client --> API
    API --> Model
    Model --> API

    API --> Prom
    Prom --> Gov

    DVC --> Train
    Train --> MLflow
    MLflow --> Deploy
    Deploy --> API

    Gov --> LLM
```
---

## 🏗️ Technical Architecture (Layered View)

```mermaid
graph TB
    subgraph Client
        User
    end

    subgraph Kubernetes
        Ingress

        subgraph Stable
            API1[FastAPI + Model]
        end

        subgraph Canary
            API2[FastAPI + Model]
        end
    end

    subgraph ML_Platform
        DVC[DVC Pipelines]
        Train[Training Job]
        MLflow[MLflow Registry]
        S3[S3 Artifacts]
    end

    subgraph Monitoring
        Prometheus
        Grafana
    end

    subgraph Agents
        DeployAgent[Deployment Agent]
        GovAgent[Governance Agent]
        LLMAgent[LLM Agent]
    end

    User --> Ingress --> API1
    User --> Ingress --> API2

    API1 --> Prometheus
    API2 --> Prometheus

    DVC --> Train --> MLflow
    MLflow --> DeployAgent
    DeployAgent --> API2
    DeployAgent --> API1

    Prometheus --> GovAgent
    GovAgent --> LLMAgent
```


---

## 🔁 End-to-End System Flow


```mermaid
sequenceDiagram
    participant Data as Raw Data
    participant DVC as DVC Pipeline
    participant Train as Training Job
    participant MLflow as MLflow Registry
    participant Deploy as Deployment Agent
    participant Canary as Canary Deployment
    participant Stable as Stable Deployment
    participant Prom as Prometheus
    participant Gov as Governance Agent
    participant LLM as LLM Agent

    Data->>DVC: Version & track data
    DVC->>Train: dvc repro
    Train->>MLflow: Register model (Staging)
    MLflow->>Deploy: Promote model
    Deploy->>Canary: Deploy new model
    Canary->>Prom: Emit latency & errors
    Stable->>Prom: Emit latency & errors
    Prom->>Gov: Metrics polling
    Gov->>LLM: Final governance decision
    LLM->>LLM: Generate JSON + PDF report
```

---

## 🧠 ML Pipeline (Offline)

Managed entirely via **DVC** for reproducibility.

```text
Raw Data
 → Preprocessing
 → Feature Engineering
 → Model Training
 → Evaluation
 → Drift Detection
 → MLflow Registry
```

---

## 🚀 Model Serving (Online)

**FastAPI-based inference service**:

* `POST /predict` — Low-latency inference
* `POST /predict/explain` — SHAP explanations
* `GET /health` — Kubernetes probes
* `GET /metrics` — Prometheus metrics

Model artifacts are **loaded once at startup** and cached in memory to minimize latency.

---

## ⚡ SLA & Performance Targets

| Metric       | Target   |
| ------------ | -------- |
| P95 Latency  | < 400 ms |
| Error Rate   | < 1%     |
| Availability | 99.9%    |



---

## 📈 Monitoring & Observability

* Prometheus scrapes `/metrics`
* Latency histograms
* Error counters
* Throughput metrics

These signals directly feed into **automated governance decisions**.

---

## 🤖 Autonomous Governance Agents

### 1️⃣ Deployment Agent

**Purpose:** Decouple training from serving

* Reads MLflow registry
* Promotes models to Production
* Triggers Kubernetes rollouts

---

### 2️⃣ Governance Agent

**Purpose:** ML SRE automation

* Monitors latency, errors, and SLA
* Compares Canary vs Stable
* Automatically decides:

  * Accept deployment
  * Reject deployment
  * Rollback
  * Scale resources

---

### 3️⃣ LLM Governance Agent

**Purpose:** Executive-level visibility

* Consumes:

  * Model metrics
  * Drift reports
  * SHAP explanations
  * Production SLAs
* Produces:

  * `deployment_report.json`
  * `deployment_report.pdf`

---

## 🔄 CI/CD Strategy

### Why Two Pipelines?

| Pipeline              | Trigger       | Responsibility   |
| --------------------- | ------------- | ---------------- |
| `inference-cicd.yaml` | Every push    | API & serving    |
| `training-cicd.yaml`  | Manual / Cron | Model retraining |

**Core Principle:**

> Inference changes often. Training should not.

---




## 🧰 Tech Stack

* **Backend:** FastAPI, Python
* **ML:** LightGBM, SHAP
* **MLOps:** DVC, MLflow
* **Infra:** Docker, AWS EKS
* **Monitoring:** Prometheus, Grafana
* **Governance:** Rule-based agents
* **LLM:** OpenAI (JSON-mode reports)

---

## 🏁 Final Takeaway

AutoML Brain is **not a single ML model**.

It is a:

> **Self-governing, production-ready ML platform**
> built for environments where **latency, safety, and governance matter as much as accuracy**.

