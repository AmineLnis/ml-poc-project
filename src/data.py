"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DATASET_FILE = (
    Path(__file__).parent.parent / "data" / "household_energy_consumption.csv"
)
TARGET_COLUMN = "Energy_Consumption_kWh"
FEATURE_COLUMNS = [
    "Household_Size",
    "Avg_Temperature_C",
    "Has_AC_Binary",
    "temperature_x_ac",
    "household_size_x_ac",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """

    df = pd.read_csv(DATASET_FILE)

    required_columns = {
        TARGET_COLUMN,
        "Household_Size",
        "Avg_Temperature_C",
        "Has_AC",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df.dropna(subset=list(required_columns)).copy()
    if len(df) < 2:
        raise ValueError("Dataset must contain at least two valid rows to split.")

    df["Has_AC_Binary"] = (
        df["Has_AC"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    if df["Has_AC_Binary"].isna().any():
        raise ValueError("Column Has_AC must contain only Yes/No values.")

    df["temperature_x_ac"] = df["Avg_Temperature_C"] * df["Has_AC_Binary"]
    df["household_size_x_ac"] = df["Household_Size"] * df["Has_AC_Binary"]

    shuffled = df.sample(frac=1, random_state=RANDOM_STATE)
    test_count = max(1, int(len(shuffled) * TEST_SIZE))
    train_df = shuffled.iloc[:-test_count]
    test_df = shuffled.iloc[-test_count:]

    X_train = train_df[FEATURE_COLUMNS].reset_index(drop=True)
    X_test = test_df[FEATURE_COLUMNS].reset_index(drop=True)
    y_train = train_df[TARGET_COLUMN].reset_index(drop=True)
    y_test = test_df[TARGET_COLUMN].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
