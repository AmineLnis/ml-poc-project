"""Evaluate configured models and optionally launch the Streamlit app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import APP_ENTRYPOINT, MODELS, STREAMLIT_HOST, STREAMLIT_PORT
from data import load_dataset_split
from metrics import compute_metrics
from model_io import load_model
from results import write_metrics


def evaluate_models() -> None:
    """Evaluate every model registered in config.MODELS on the shared test split."""

    _, X_test, _, y_test = load_dataset_split()
    rows: list[dict[str, object]] = []

    for model_key, model_config in MODELS.items():
        model = load_model(model_config["path"])
        y_pred = model.predict(X_test)
        rows.append({"model": model_key, **compute_metrics(y_test, y_pred)})

    metrics_df = write_metrics(rows).sort_values("rmse").reset_index(drop=True)
    write_metrics(metrics_df.to_dict(orient="records"))
    print(metrics_df.to_string(index=False))


def launch_streamlit() -> None:
    """Launch the Streamlit app with the project src directory on PYTHONPATH."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_ENTRYPOINT),
            "--server.address",
            STREAMLIT_HOST,
            "--server.port",
            str(STREAMLIT_PORT),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


def main(argv: list[str] | None = None) -> None:
    """Run model evaluation, then launch the Streamlit app unless disabled."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Evaluate models without launching Streamlit.",
    )
    args = parser.parse_args(argv)

    evaluate_models()
    if not args.no_app:
        launch_streamlit()


if __name__ == "__main__":
    main()
