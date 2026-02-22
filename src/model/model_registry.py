import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
import dagshub


dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="AutoML-Brain",
    mlflow=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_registry")

MODEL_NAME = "churn_model"


def main():

    client = MlflowClient()

    logger.info("Loading run_information.json")

    with open("run_information.json") as f:
        run_info = json.load(f)

    model_uri = run_info["model_uri"]

    logger.info(f"Registering model from URI: {model_uri}")

    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_info["run_id"]
    )

    logger.info(
        f"Model registered as {MODEL_NAME} v{model_version.version}"
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True
    )

    logger.info(
        f"Model v{model_version.version} moved to Staging"
    )


if __name__ == "__main__":
    main()