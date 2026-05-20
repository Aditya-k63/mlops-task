# Minimal MLOps Batch Job

This project is a small batch processing pipeline built for the ML/MLOps Engineering Internship technical assesment.

The program reads market OHLCV data from a CSV file, computes a rolling mean on the `close` price, generates a simple binary trading signal, and saves metrics in JSON format. It also writes detailed logs so the whole execution can be tracked easily.

The project is deterministic because the random seed is loaded from a YAML config file.

---

## What This Project Does

The script performs the following steps:

1. Loads configuration from `config.yaml`
2. Validates required config fields
3. Reads `data.csv`
4. Computes rolling mean on the `close` column
5. Generates signal:

   * `1` if close > rolling mean
   * `0` otherwise
6. Calculates summary metrics
7. Saves metrics to `metrics.json`
8. Writes logs to `run.log`

---

## Project Structure

```text
mlops-task/
├── .github/
│   └── workflows/
│       └── ci.yml
├── run.py
├── config.yaml
├── data.csv
├── requirements.txt
├── Dockerfile
├── README.md
├── metrics.json
└── run.log
```

---

## Config File

The `config.yaml` file contains:

```yaml
seed: 42
window: 5
version: "v1"
```

* `seed` is used to make the run reproducable.
* `window` is the rolling mean window size.
* `version` is included in the final metrics output.

---

## How to Run Locally

First install the required packages:

```bash
pip install -r requirements.txt
```

Then run the script:

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

---

## Example Output

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 54,
  "seed": 42,
  "status": "success"
}
```

---

## Logging

All execution steps are written to `run.log`, including:

* Job start and end
* Config validation
* Dataset loading
* Rolling mean calculation
* Signal generation
* Final metrics
* Any errors if they occur

---

## Error Handling

The script handles several possible issues:

* Missing config file
* Missing input CSV
* Empty dataset
* Missing `close` column
* Invalid YAML structure
* Encoding and delimiter issues in CSV

Even if the job fails, a `metrics.json` file is still generated with an error message.

---

## Docker Usage

### Build Docker Image

```bash
docker build -t mlops-task .
```

### Run Docker Container

```bash
docker run --rm mlops-task
```

The container prints the final metrics to stdout and generates `metrics.json` and `run.log`.

---

## CI/CD with GitHub Actions

This project is automatically tested using GitHub Actions.

The workflow does the following:

* Installs dependencies
* Runs the batch job
* Uploads generated artifacts
* Builds Docker image
* Runs the Docker container

This was useful because my local system is not very powerfull for Docker, so I used GitHub Actions to validate everything in the cloud.

---

## Design Decisions

* Rolling mean is computed using pandas.
* The first few rows where rolling mean is NaN produce signal `0`.
* CSV loading is made more robust to handle different encodings and unusual separators.
* Metrics are always written, even in failure cases.

---

## Technologies Used

* Python
* Pandas
* NumPy
* PyYAML
* Docker
* GitHub Actions

---

## Notes

This project is intentionally simple, but it follows some important MLOps practices like reproducibility, observability, containerization, and automated testing.

I tried to keep the code clean and easy to understand while still handling real-world edge cases.
