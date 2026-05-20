import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml


DEFAULT_VERSION = "v1"


def setup_logger(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def write_metrics(output_path: str, metrics: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure. Expected YAML dictionary.")

    required_fields = ["seed", "window", "version"]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValueError("window must be a positive integer")

    if not isinstance(config["version"], str):
        raise ValueError("version must be a string")

    return config


def find_header_row(input_path: str, encoding: str) -> int:
    """Return the 0-based index of the row containing OHLCV headers."""
    with open(input_path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if "close" in line.lower():
                return i
    return 0  # fall back to first row if not found


def load_dataset(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    encodings = ["utf-8", "latin1", "cp1252"]
    df = None
    last_error = None

    for encoding in encodings:
        try:
            logging.info("Trying to read CSV with encoding=%s", encoding)

            # Read using semicolon separator
            df = pd.read_csv(
                input_path,
                encoding=encoding,
                sep=";",
                engine="python"
            )

            logging.info("Successfully read CSV with encoding=%s", encoding)
            break

        except Exception as e:
            last_error = e
            continue

    if df is None:
        raise ValueError(f"Invalid CSV format: {last_error}")

    if df.empty:
        raise ValueError("Input CSV is empty")

    # Split the first column if header was read as a single string
    if len(df.columns) >= 1 and "close" not in [col.lower().strip() for col in df.columns]:
        first_col = df.columns[0]

        if "," in first_col:
            logging.info("Detected embedded comma-separated header. Reparsing first column.")

            # Split the first column into multiple columns
            expanded = df[first_col].str.split(",", expand=True)

            # Set correct column names
            expanded.columns = [col.strip().lower() for col in first_col.split(",")]

            df = expanded

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    # Convert close to numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric values found in 'close' column")

    return df


def process_data(df: pd.DataFrame, window: int) -> pd.DataFrame:
    logging.info("Computing rolling mean with window=%d", window)
    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    logging.info("Generating binary signal")
    # NaN comparisons evaluate to False, which becomes 0.
    df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)

    return df


def compute_metrics(df: pd.DataFrame, config: dict, latency_ms: int) -> dict:
    signal_rate = float(df["signal"].mean())

    return {
        "version": config["version"],
        "rows_processed": int(len(df)),
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": int(latency_ms),
        "seed": config["seed"],
        "status": "success",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal MLOps batch job")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def main():
    args = parse_args()

    setup_logger(args.log_file)
    logging.info("Job started")

    start_time = time.perf_counter()
    version = DEFAULT_VERSION

    try:
        logging.info("Loading configuration from %s", args.config)
        config = load_config(args.config)
        version = config["version"]

        logging.info(
            "Config validated: seed=%s, window=%s, version=%s",
            config["seed"],
            config["window"],
            config["version"],
        )

        np.random.seed(config["seed"])

        logging.info("Loading dataset from %s", args.input)
        df = load_dataset(args.input)
        logging.info("Rows loaded: %d", len(df))

        df = process_data(df, config["window"])

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        metrics = compute_metrics(df, config, latency_ms)

        logging.info("Metrics summary: %s", metrics)

        write_metrics(args.output, metrics)

        logging.info("Job completed successfully")
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as e:
        logging.exception("Job failed")

        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(e),
        }

        try:
            write_metrics(args.output, error_metrics)
        except Exception:
            pass

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()