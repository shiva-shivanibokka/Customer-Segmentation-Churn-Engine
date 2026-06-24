# Architecture

## System Overview

The Customer Segmentation & Churn Engine is an **offline batch + on-demand serving** system. The pipeline runs once to produce all model artifacts, which are then served by the Streamlit dashboard and the FastAPI scoring endpoint without re-training.

```
Raw Excel (data/raw/)
       │
       ▼
┌─────────────────┐
│  features.py    │  Schema validation → imputation → encoding → feature engineering
└────────┬────────┘
         │ features.parquet
         ▼
┌─────────────────┐
│ segmentation.py │  K-Means++ → GMM soft probabilities → UMAP → bootstrap ARI stability
└────────┬────────┘
         │ segmented.parquet + kmeans.pkl + gmm.pkl + scaler.pkl
         ▼
┌─────────────────┐
│ churn_model.py  │  Per-segment XGBoost → isotonic calibration → MLflow tracking
└────────┬────────┘
         │ scored.parquet + segment_models.pkl
         ▼
┌─────────────────┐
│ uplift_model.py │  T-Learner + S-Learner (CausalML) → ROI ranking
└────────┬────────┘
         │ uplift.parquet + uplift_metrics.pkl
         ▼
  ┌──────┴──────┐
  │             │
  ▼             ▼
app.py       api/serve.py
(Streamlit)  (FastAPI)
```

---

## Architecture Decision Records

### ADR-1: Offline Batch Pipeline, Not Real-Time Feature Computation

**Decision:** All feature engineering, model training, and scoring runs in a batch pipeline (`pipeline.py`) and outputs cached Parquet/pkl files. The Streamlit app and API read from disk — they never re-train or re-engineer features at request time.

**Alternatives considered:**
- Real-time feature computation on every API request.
- An online feature store (e.g., Redis, Feast).

**Reasoning:**
- The dataset is static (5,630 customers from a snapshot). There is no streaming data source to update from.
- Batch materialization mirrors how production feature stores (Uber Michelangelo, Airbnb Chronon) separate offline training from online serving: features are computed once and looked up at inference time.
- For a portfolio system this removes the infrastructure dependency on a running feature store, keeping the setup to `pip install + python src/pipeline.py`.

**Trade-off:** The system cannot score customers whose behavioral data changes between pipeline runs. At real production scale, you would replace the batch pipeline with a streaming feature pipeline (Kafka → Flink → feature store) and set up a retraining trigger when feature drift exceeds a threshold.

---

### ADR-2: Parquet Files Over a Relational Database

**Decision:** Processed features and model outputs are stored as Parquet files in `data/processed/`. Models are stored as Pickle files in `models/`.

**Alternatives considered:**
- SQLite (zero-config relational DB).
- PostgreSQL (full RDBMS).
- DuckDB (columnar SQL over files).

**Reasoning:**
- Parquet is a columnar format — reading a 30-column, 5,630-row dataset takes milliseconds and requires no database server to be running.
- The access pattern is always "read entire table" or "filter by Segment" — both are optimal in columnar format and inefficient to over-engineer with a B-tree index.
- Eliminates the database dependency for local setup, Docker, and Streamlit Community Cloud (no persistent volume required).

**Trade-off:** No support for incremental updates (appending new customers without rewriting the file), transactions, or concurrent writes. At 10x scale with daily customer data appends, you would move to a Delta Lake or Iceberg table format for ACID incremental writes, or a proper data warehouse (BigQuery, Redshift).

---

### ADR-3: Per-Segment Models Over a Single Global Model

**Decision:** Five separate XGBoost classifiers are trained — one per behavioral segment — rather than a single model trained on all customers.

**Alternatives considered:**
- One global XGBoost model with Segment as a feature.
- One global model with segment interaction terms.

**Reasoning:**
- A Champion churns for fundamentally different reasons than a Lapsed customer. Champions who churn usually have a discrete trigger (bad support experience, competitor offer). Lapsed customers churn through gradual disengagement. A single model that sees both groups in the same training set will regress to the average behavior and underfit both.
- This mirrors Salesforce Einstein's per-tier health scoring, where separate models are trained per customer tier.
- The CV AUC range across segments (0.945–0.984) confirms segments have meaningfully different churn dynamics — a single global model would have averaged these out.

**Trade-off:** Five models to maintain, calibrate, and version instead of one. Small segments (< 50 samples) are skipped. At scale, you would handle small segments via transfer learning from the global model (warm-starting per-segment fine-tuning) rather than training from scratch.

---

### ADR-4: File-Based Model Cache Over a Model Registry

**Decision:** Trained models are saved as `.pkl` files in `models/`. The pipeline checks for existing files before re-training (`force_retrain=False` default).

**Alternatives considered:**
- MLflow Model Registry (already used for experiment tracking).
- A dedicated model store (Seldon, BentoML, SageMaker Model Registry).

**Reasoning:**
- MLflow experiment tracking is already wired up for metrics and artifact logging. Promoting to a full registry adds operational overhead (a running MLflow server) that isn't warranted for a single-environment deployment.
- The file cache provides reproducibility without a network dependency: `models/segment_models.pkl` will produce identical inference results to the tracked MLflow run.
- Streamlit Community Cloud has no persistent storage beyond the repo itself — file-based caching is the only viable option in that environment.

**Trade-off:** No automated model promotion, A/B testing infrastructure, or rollback-by-version. At production scale, you would use the MLflow Model Registry's `Production` / `Staging` / `Archived` lifecycle stages and serve via a dedicated inference server that can hot-swap model versions.

---

## What Changes at 10× Scale (50,000+ customers, daily updates)

| Component | Current | At 10× scale |
|---|---|---|
| Feature pipeline | Pandas batch on a static file | Spark or dbt on a data warehouse; Kafka for streaming behavioral events |
| Feature storage | Parquet files | Feature store (Feast, Tecton) with point-in-time correctness |
| Model training | Single run on full dataset | Incremental training trigger on feature drift; MLflow Model Registry promotion workflow |
| Model serving | FastAPI on one process | Multiple FastAPI replicas behind a load balancer; model loaded from registry not disk |
| Experiment tracking | Local MLflow runs | MLflow on a dedicated server or hosted (Databricks) |
| Monitoring | Structured logs | PSI/KS drift detection on feature distributions; Grafana dashboard on inference latency |
