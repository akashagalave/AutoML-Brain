from kubernetes import client, config
import time


def main():
    config.load_incluster_config()

    batch_v1 = client.BatchV1Api()

    unique_name = f"deployment-agent-trigger-{int(time.time())}"

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": unique_name,
            "namespace": "automl-brain",
        },
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "serviceAccountName": "deployment-agent-sa",
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "deployment-agent",
                            "image": "199964507152.dkr.ecr.us-east-1.amazonaws.com/automl-brain-deployment-agent:latest",
                            "imagePullPolicy": "Always",
                        }
                    ],
                }
            },
        },
    }

    batch_v1.create_namespaced_job(
        namespace="automl-brain",
        body=job_manifest
    )

    print(f"Deployment trigger job {unique_name} created.")


if __name__ == "__main__":
    main()