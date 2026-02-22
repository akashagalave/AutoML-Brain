from kubernetes import client, config
import time
import logging

# ===== Retrain Protection Config =====
RETRAIN_COOLDOWN_SECONDS = 3600  # 1 hour safety cooldown

_last_retrain_time = 0


def get_current_replicas(namespace: str, deployment: str) -> int:
    config.load_incluster_config()
    api = client.AppsV1Api()
    dep = api.read_namespaced_deployment(
        name=deployment,
        namespace=namespace
    )
    return dep.spec.replicas


def scale_deployment(namespace: str, deployment: str, replicas: int):
    config.load_incluster_config()
    api = client.AppsV1Api()

    body = {
        "spec": {
            "replicas": replicas
        }
    }

    api.patch_namespaced_deployment_scale(
        name=deployment,
        namespace=namespace,
        body=body
    )


def retrain_job_running(namespace: str) -> bool:
    """
    Checks if any retrain job is currently active.
    Prevents duplicate retrain storms.
    """
    config.load_incluster_config()
    batch_api = client.BatchV1Api()

    jobs = batch_api.list_namespaced_job(namespace=namespace)

    for job in jobs.items:
        if job.metadata.name.startswith("automl-retrain-"):
            if job.status.active:
                return True

    return False


def trigger_retraining_job(namespace: str):
    """
    Safe retraining trigger with:
    - Cooldown protection
    - Active job detection
    """
    global _last_retrain_time

    now = time.time()

    # Cooldown guard
    if now - _last_retrain_time < RETRAIN_COOLDOWN_SECONDS:
        logging.warning("Retrain skipped — cooldown active.")
        return

    # Running job guard
    if retrain_job_running(namespace):
        logging.warning("Retrain skipped — job already running.")
        return

    config.load_incluster_config()
    batch_api = client.BatchV1Api()

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": "automl-retrain-"
        },
        "spec": {
            "backoffLimit": 1,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "spec": {
                    "serviceAccountName": "retrain-sa",
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "retraining",
                            "image": "199964507152.dkr.ecr.us-east-1.amazonaws.com/automl-brain-training:latest",
                            "imagePullPolicy": "Always",
                            "env": [
                                {
                                    "name": "AWS_REGION",
                                    "value": "us-east-1"
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }

    batch_api.create_namespaced_job(
        namespace=namespace,
        body=job_manifest
    )

    _last_retrain_time = now
    logging.error("Autonomous retraining job triggered safely.")