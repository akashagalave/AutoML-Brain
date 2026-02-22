import logging
import dagshub
import mlflow
from mlflow.tracking import MlflowClient


# -----------------------------
# Init
# -----------------------------

dagshub.init(
    repo_owner="akashagalaveaaa1",
    repo_name="AutoML-Brain",
    mlflow=True
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auto_promote")

MODEL_NAME = "churn_model"
METRIC_NAME = "roc_auc"

SOURCE_STAGE = "Staging"
TARGET_STAGE = "Production"


def get_metric_from_run(client, run_id, metric_name):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_name)


def main():

    client = MlflowClient()

    # -----------------------------
    # Get latest Staging model
    # -----------------------------

    staging_versions = client.get_latest_versions(
        name=MODEL_NAME,
        stages=[SOURCE_STAGE]
    )

    if not staging_versions:
        logger.info("No model found in Staging.")
        return

    staging_model = staging_versions[0]
    staging_run_id = staging_model.run_id

    staging_metric = get_metric_from_run(
        client,
        staging_run_id,
        METRIC_NAME
    )

    logger.info(f"Staging ROC-AUC: {staging_metric}")

    # -----------------------------
    # Get current Production model
    # -----------------------------

    production_versions = client.get_latest_versions(
        name=MODEL_NAME,
        stages=[TARGET_STAGE]
    )

    if not production_versions:
        logger.info("No Production model found. Promoting directly.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_model.version,
            stage=TARGET_STAGE,
            archive_existing_versions=True
        )
        logger.info("Promoted to Production.")
        return

    production_model = production_versions[0]
    production_run_id = production_model.run_id

    production_metric = get_metric_from_run(
        client,
        production_run_id,
        METRIC_NAME
    )

    logger.info(f"Production ROC-AUC: {production_metric}")

    # -----------------------------
    # Compare
    # -----------------------------

    if staging_metric > production_metric:
        logger.info("New model is better. Promoting to Production.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_model.version,
            stage=TARGET_STAGE,
            archive_existing_versions=True
        )

        logger.info("Promotion completed.")

    else:
        logger.info("New model is NOT better. No promotion performed.")


if __name__ == "__main__":
    main()