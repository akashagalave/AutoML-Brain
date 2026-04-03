import logging
import os
import time
import requests
from kubernetes import client, config
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deployment_agent")

MODEL_NAME = "churn_model"
NAMESPACE = "automl-brain"
STABLE_DEPLOYMENT = "inference-stable"
CANARY_DEPLOYMENT = "inference-canary"

PROMETHEUS_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"

WAIT_SECONDS = 300
MAX_ALLOWED_P95 = 0.6
RELATIVE_THRESHOLD = 1.10


def query_prometheus(query):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=10,
    )
    data = response.json()

    if data["status"] != "success":
        raise Exception("Prometheus query failed")

    result = data["data"]["result"]
    if not result:
        return None

    return float(result[0]["value"][1])


def get_p95(track):
    if track == "canary":
        pod_regex = "inference-canary.*"
    else:
        pod_regex = "inference-stable.*"

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

    return query_prometheus(query)


def get_staging_version():
    """Return latest Staging version object without promoting it."""
    mlflow_client = MlflowClient()
    versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_versions = [v for v in versions if v.current_stage == "Staging"]

    if not staging_versions:
        return None

    return sorted(staging_versions, key=lambda x: int(x.version), reverse=True)[0]


def promote_model_to_production(version_obj):
    """Promote a specific already-identified staging version to Production."""
    mlflow_client = MlflowClient()

    logger.info(f"Promoting model version {version_obj.version} to Production")

    mlflow_client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_obj.version,
        stage="Production",
        archive_existing_versions=True,
    )

    return version_obj.version


def rollback_model_to_staging(version_obj):

    mlflow_client = MlflowClient()

    logger.warning(
        f"Rolling back model version {version_obj.version} from Production → Staging"
    )

    mlflow_client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version_obj.version,
        stage="Staging",
        archive_existing_versions=False,
    )


def restart_canary_deployment():

    logger.info("Restarting canary deployment to load latest Staging model...")

    config.load_incluster_config()
    apps_v1 = client.AppsV1Api()

    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "canary-restart": str(os.urandom(4))
                    }
                }
            }
        }
    }

    apps_v1.patch_namespaced_deployment(
        name=CANARY_DEPLOYMENT,
        namespace=NAMESPACE,
        body=body,
    )

    logger.info("Canary deployment restarted.")


def restart_stable_deployment():
    logger.info("Restarting stable deployment...")

    config.load_incluster_config()
    apps_v1 = client.AppsV1Api()

    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "model-restart": str(os.urandom(4))
                    }
                }
            }
        }
    }

    apps_v1.patch_namespaced_deployment(
        name=STABLE_DEPLOYMENT,
        namespace=NAMESPACE,
        body=body,
    )

    logger.info("Stable deployment restarted.")


def evaluate_canary():
    logger.info(f"Waiting {WAIT_SECONDS} seconds for canary metrics...")
    time.sleep(WAIT_SECONDS)

    canary_p95 = get_p95("canary")
    stable_p95 = get_p95("stable")

    logger.info(f"Canary p95: {canary_p95}")
    logger.info(f"Stable p95: {stable_p95}")

    if canary_p95 is None or stable_p95 is None:
        logger.warning("Insufficient metrics data.")
        return False

    if canary_p95 > MAX_ALLOWED_P95:
        logger.warning("Canary exceeds absolute latency threshold.")
        return False

    if canary_p95 > stable_p95 * RELATIVE_THRESHOLD:
        logger.warning("Canary exceeds relative latency threshold.")
        return False

    logger.info("Canary passed evaluation.")
    return True


def main():
    staging_version = get_staging_version()
    if not staging_version:
        logger.info("No staging model found. Nothing to promote.")
        return

    restart_canary_deployment()

    version = promote_model_to_production(staging_version)

    safe = evaluate_canary()

    if safe:
 
        restart_stable_deployment()
        logger.info(
            f"Model version {version} promoted and stable deployment updated."
        )
    else:

        rollback_model_to_staging(staging_version)
        logger.warning(
            "Canary failed evaluation. MLflow rolled back to Staging. "
            "Stable deployment untouched."
        )


if __name__ == "__main__":
    main()