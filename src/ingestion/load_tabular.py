"""
Data ingestion for the UCI Adult Income dataset.

Downloads the dataset directly from the UCI ML Repository (if not already
cached locally), cleans it, encodes categorical features, and returns
train/val/test splits as specified in the TAAS proposal (70/15/15).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

UCI_ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

RAW_PATH = "data/raw/adult.csv"
PROCESSED_PATH = "data/processed/adult_processed.csv"


def _download_from_uci() -> pd.DataFrame:
    return pd.read_csv(
        UCI_ADULT_URL,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
    )


def _download_from_openml() -> pd.DataFrame:
    """Fallback: fetch the same dataset via sklearn's OpenML mirror."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"class": "income"})
    df["income"] = df["income"].astype(str).str.replace("b'", "").str.replace("'", "")
    return df


def download_adult_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Download the UCI Adult Income dataset if not already cached locally.
    Tries the direct UCI URL first, falls back to the OpenML mirror if that
    fails (network restrictions, UCI rate-limiting, or URL changes).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return pd.read_csv(path)

    try:
        df = _download_from_uci()
    except Exception as e:
        print(f"UCI direct download failed ({e}); falling back to OpenML mirror...")
        df = _download_from_openml()

    df.to_csv(path, index=False)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Clean and encode the dataset.

    Returns:
        X: feature dataframe (numeric, model-ready)
        y: binary target (1 if income > 50K else 0)
        encoders: dict of fitted LabelEncoders per categorical column,
                  needed later to decode SHAP feature explanations.
    """
    df = df.dropna().reset_index(drop=True)

    y = (df["income"].str.strip() == ">50K").astype(int)
    X = df.drop(columns=["income"])

    encoders = {}
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders


def get_splits(test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test) using a
    70/15/15 split, matching the proposal's stated methodology.
    """
    df = download_adult_dataset()
    X, y, encoders = preprocess(df)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, _ = get_splits()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")
