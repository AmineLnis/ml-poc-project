"""Launch the Streamlit app and optionally evaluate configured models."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import APP_ENTRYPOINT, MODELS, STREAMLIT_HOST, STREAMLIT_PORT


def evaluate_models() -> None:
    """Evaluate every model registered in config.MODELS on the shared test split."""

    from data import load_dataset_split
    from metrics import compute_metrics
    from model_io import load_model
    from results import write_metrics

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
    port = find_available_port(STREAMLIT_HOST, STREAMLIT_PORT)
    if port != STREAMLIT_PORT:
        print(f"Port {STREAMLIT_PORT} indisponible, lancement sur le port {port}.")

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
            str(port),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )


def find_available_port(host: str, preferred_port: int) -> int:
    """Return the preferred port, or the next available port."""

    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Aucun port disponible entre {preferred_port} et {preferred_port + 19}."
    )


def main(argv: list[str] | None = None) -> None:
    """Launch Streamlit by default, with optional model evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate models before launching Streamlit.",
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="Evaluate models without launching Streamlit. Implies --evaluate.",
    )
    args = parser.parse_args(argv)

    if args.evaluate or args.no_app:
        evaluate_models()
    if not args.no_app:
        launch_streamlit()


if __name__ == "__main__":
    main()
