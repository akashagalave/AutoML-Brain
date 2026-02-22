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

CPU_QUERY = """
avg(
  rate(
    container_cpu_usage_seconds_total{
      namespace="automl-brain",
      pod=~"inference-deployment.*",
      container!="POD"
    }[2m]
  )
)
"""

ERROR_RATE_QUERY = """
(
  sum(
    rate(
      http_requests_total{
        namespace="automl-brain",
        status=~"5.."
      }[2m]
    )
  )
)
/
(
  sum(
    rate(
      http_requests_total{
        namespace="automl-brain"
      }[2m]
    )
  )
)
"""