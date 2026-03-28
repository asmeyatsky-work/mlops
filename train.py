"""
Default training script for Vertex AI CustomTrainingJob.

This script is referenced by vertex_training_adapter.py as script_path="train.py".
It runs inside the training container on Vertex AI.

Environment variables set by Vertex AI:
    AIP_MODEL_DIR       — GCS path to write the trained model artifact
    AIP_DATA_FORMAT     — Format of the training data (e.g., "csv", "bigquery")
    AIP_TRAINING_DATA_URI — GCS URI or BQ URI for training data
"""
from __future__ import annotations

import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    model_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    data_uri = os.environ.get("AIP_TRAINING_DATA_URI", "")
    data_format = os.environ.get("AIP_DATA_FORMAT", "unknown")

    logger.info("Starting training")
    logger.info("  Model output dir: %s", model_dir)
    logger.info("  Data URI: %s", data_uri)
    logger.info("  Data format: %s", data_format)

    # --- Load data ---
    # In a real implementation, load from data_uri based on data_format.
    # For BigQuery: use google.cloud.bigquery
    # For GCS CSV: use pandas + gcsfs
    logger.info("Loading training data...")

    # --- Train model ---
    # Replace with actual training logic (sklearn, tf, pytorch, xgboost, etc.)
    logger.info("Training model...")
    model_metadata = {
        "framework": "placeholder",
        "data_uri": data_uri,
        "data_format": data_format,
        "metrics": {
            "accuracy": 0.0,
            "f1_score": 0.0,
        },
    }

    # --- Save model ---
    os.makedirs(model_dir, exist_ok=True)
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)

    logger.info("Model saved to %s", model_dir)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
