PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"

P95_QUERY = """
histogram_quantile(
  0.95,
  sum(
    rate(
      churn_request_latency_seconds_bucket{path="/predict"}[5m]
    )
  ) by (le)
)
"""

LATENCY_THRESHOLD = 0.4
LOW_LATENCY_THRESHOLD = 0.2

BREACH_CYCLES = 3
RECOVERY_CYCLES = 5

POLL_INTERVAL = 30
COOLDOWN_SECONDS = 120

MIN_REPLICAS = 2
MAX_REPLICAS = 6
SCALE_STEP = 1

TARGET_NAMESPACE = "automl-brain"
TARGET_DEPLOYMENT = "inference-deployment"