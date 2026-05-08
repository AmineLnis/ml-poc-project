"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any

import numpy as np


METRIC_NAMES = ("mae", "mse", "rmse", "r2")


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """

    y_true_array = _to_1d_float_array(y_true, "y_true")
    y_pred_array = _to_1d_float_array(y_pred, "y_pred")

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true_array.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    errors = y_true_array - y_pred_array
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    mae = float(np.mean(absolute_errors))
    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    r2 = _compute_r2(y_true_array, squared_errors)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def _to_1d_float_array(values: Any, name: str) -> np.ndarray:
    """Convert model outputs to a validated 1D numeric array."""

    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values.")

    return array


def _compute_r2(y_true_array: np.ndarray, squared_errors: np.ndarray) -> float:
    """Compute R2 with a finite result for constant targets."""

    ss_res = float(np.sum(squared_errors))
    ss_tot = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1 - (ss_res / ss_tot))
