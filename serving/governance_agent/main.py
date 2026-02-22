import time
import logging

from prometheus_client import query_prometheus
from rule_engine import GovernanceState
from k8s_controller import (
    scale_deployment,
    get_current_replicas,
    trigger_retraining_job
)
from metrics_queries import P95_QUERY, CPU_QUERY, ERROR_RATE_QUERY
from config import (
    POLL_INTERVAL,
    TARGET_NAMESPACE,
    TARGET_DEPLOYMENT,
    MIN_REPLICAS,
    MAX_REPLICAS,
    SCALE_STEP
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():
    logging.info("Governance Agent v5 (Safe Autonomous Mode) started.")
    state = GovernanceState()

    while True:
        try:
            p95 = query_prometheus(P95_QUERY)
            cpu = query_prometheus(CPU_QUERY)
            error_rate = query_prometheus(ERROR_RATE_QUERY)

            logging.info(
                f"Signals → p95: {p95}, CPU: {cpu}, ErrorRate: {error_rate}"
            )

            action = state.evaluate(p95, cpu, error_rate)

            if action:
                current = get_current_replicas(
                    TARGET_NAMESPACE,
                    TARGET_DEPLOYMENT
                )

                if action == "scale_up" and current < MAX_REPLICAS:
                    new_replicas = min(current + SCALE_STEP, MAX_REPLICAS)
                    logging.warning(f"Scaling UP from {current} to {new_replicas}")
                    scale_deployment(
                        TARGET_NAMESPACE,
                        TARGET_DEPLOYMENT,
                        new_replicas
                    )

                elif action == "scale_down" and current > MIN_REPLICAS:
                    new_replicas = max(current - SCALE_STEP, MIN_REPLICAS)
                    logging.warning(f"Scaling DOWN from {current} to {new_replicas}")
                    scale_deployment(
                        TARGET_NAMESPACE,
                        TARGET_DEPLOYMENT,
                        new_replicas
                    )

                elif action == "retrain":
                    logging.error("SLA violation detected — evaluating retrain policy.")
                    trigger_retraining_job(TARGET_NAMESPACE)

        except Exception as e:
            logging.error(f"Error in governance loop: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()