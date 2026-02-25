import logging
import dagshub
from mlflow.tracking import MlflowClient



dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="AutoML-Brain",
    mlflow=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("promote_model")

MODEL_NAME = "churn_model"
SOURCE_STAGE = "Staging"
TARGET_STAGE = "Production"


def main():
    client = MlflowClient()

    logger.info(f"Fetching latest model from stage: {SOURCE_STAGE}")

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    staging_versions = [
        v for v in versions if v.current_stage == SOURCE_STAGE
    ]

    if not staging_versions:
        raise RuntimeError(
            f"No model found in stage '{SOURCE_STAGE}'"
        )

    latest_version = sorted(
        staging_versions,
        key=lambda x: int(x.version),
        reverse=True
    )[0]

    logger.info(
        f"Promoting model v{latest_version.version} "
        f"from {SOURCE_STAGE} to {TARGET_STAGE}"
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version.version,
        stage=TARGET_STAGE,
        archive_existing_versions=True
    )

    logger.info(
        f"Model '{MODEL_NAME}' v{latest_version.version} "
        f"promoted to {TARGET_STAGE}"
    )


if __name__ == "__main__":
    main()