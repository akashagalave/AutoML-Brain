import pandas as pd
import logging
from pathlib import Path



logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)



def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    logger.info("Basic cleaning completed.")
    return df



def split_data(df: pd.DataFrame):

    train = df[df["tenure"] <= 24]
    validation = df[(df["tenure"] > 24) & (df["tenure"] <= 48)]
    test = df[df["tenure"] > 48]

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {validation.shape}")
    logger.info(f"Test shape: {test.shape}")

    return train, validation, test



def save_splits(train, validation, test, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    logger.info("Saved train, validation, test splits.")




def main():
    root = Path(__file__).parent.parent.parent

    input_path = root / "data" / "external" / "Chrun.csv"
    output_dir = root / "data" / "raw"

    df = load_data(input_path)
    df = clean_data(df)

    train, validation, test = split_data(df)
    save_splits(train, validation, test, output_dir)

    logger.info("Data ingestion stage completed successfully.")


if __name__ == "__main__":
    main()