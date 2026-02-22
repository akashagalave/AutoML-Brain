from kubernetes import client, config

def get_current_replicas(namespace: str, deployment: str) -> int:
    config.load_incluster_config()
    api = client.AppsV1Api()
    dep = api.read_namespaced_deployment(name=deployment, namespace=namespace)
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