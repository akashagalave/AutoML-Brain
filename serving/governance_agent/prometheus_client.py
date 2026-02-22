import requests
from config import PROMETHEUS_URL

def query_prometheus(query: str):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": query},
        timeout=5
    )

    response.raise_for_status()
    data = response.json()

    if not data["data"]["result"]:
        return None

    value = float(data["data"]["result"][0]["value"][1])
    return value