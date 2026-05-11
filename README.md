# Customer Segmentation & Churn Engine

A **decision intelligence platform** for customer retention — built to mirror the architecture used by Uber, Netflix, Salesforce, and HubSpot in production. Not just a churn model: a full closed-loop system that segments customers, predicts churn per cohort, identifies which customers will *respond* to intervention (uplift modeling), and generates personalized retention strategies via LLM.

## What Makes This Different

Most churn projects do: `features → model → churn probability → send email to everyone above 0.7`

This system does what production systems actually do:

```
behavioral data
  → 8 engineered features (engagement score, recency decay, stickiness index, spend trend)
  → K-Means++ segmentation + GMM soft assignments
  → bootstrap stability validation (ARI across 100 resamplings)
  → per-segment XGBoost classifiers (separate model per cohort)
  → isotonic probability calibration (for ROI calculations)
  → uplift modeling (CausalML T-Learner + S-Learner)
  → intervention ROI ranking (who is worth spending on)
  → Claude retention action generator (what to do, through which channel, and why)
```

## The Uplift Difference

A naive churn model wastes budget on:
- **Sure Things** — would stay anyway
- **Lost Causes** — will churn regardless of intervention
- **Sleeping Dogs** — would stay but intervention triggers churn

The only valuable targets are **Persuadables** — customers who both have high churn risk AND will respond to intervention. Uber open-sourced `CausalML` for exactly this. This project uses it.

**Result: 710 Persuadables identified out of 948 high-risk customers** — avoiding wasted spend on 238 Lost Causes and 4,151 Sleeping Dogs.

## Architecture

| Stage | What | How |
|---|---|---|
| Feature Engineering | 8 behavioral composite features | pandas, domain knowledge |
| Segmentation | 5 behavioral cohorts | K-Means++, GMM, UMAP |
| Stability Validation | Bootstrap ARI (100 resamplings) | sklearn ARI |
| Churn Prediction | Per-segment binary classifiers | XGBoost + isotonic calibration |
| Explainability | Feature importance per segment | XGBoost gain scores |
| Uplift Modeling | Persuadable identification | CausalML T-Learner + S-Learner |
| ROI Ranking | Intervention priority list | CLV × uplift − cost |
| Retention Actions | Personalized CSM playbooks | Anthropic Claude |
| Experiment Tracking | Per-segment model runs | MLflow |
| UI | 4-page interactive dashboard | Streamlit |

## Model Results

| Segment | Customers | Churn Rate | CV AUC |
|---|---|---|---|
| At-Risk | 1,228 | 23.1% | 0.982 |
| Lapsed | 1,482 | 6.3% | 0.974 |
| Price Sensitive | 889 | 40.4% | 0.945 |
| Champions | 734 | 12.8% | 0.976 |
| Loyal Customers | 1,297 | 9.0% | 0.984 |

## Industry Parallels

| This Project | Production System |
|---|---|
| Per-segment XGBoost with calibration | Salesforce Einstein customer health scoring |
| Uplift T-Learner + S-Learner (CausalML) | Uber's production retention campaign targeting |
| Bootstrap ARI stability validation | Production ML segment validation |
| GMM soft probability assignments | Ambiguous health score boundary handling |
| LLM retention action generator | Salesforce Einstein Copilot CSM playbooks |
| UMAP behavioral space visualization | Netflix member segment exploration |

## Dataset

**E-Commerce Customer Churn** (Kaggle — Ankitverma2010)  
5,630 customers × 20 behavioral features including:
- Hours spent on app (engagement depth)
- Days since last order (recency signal)
- Order count, cashback, coupon usage (behavioral patterns)
- Satisfaction score, complaint history (support risk)
- Device count, address count (stickiness)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
kaggle datasets download -d ankitverma2010/ecommerce-customer-churn-analysis-and-prediction \
    -p data/raw --unzip

# 3. Run the full pipeline (builds all models and cached artifacts)
python src/pipeline.py

# 4. Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
Customer-Segmentation-Churn-Engine/
├── app.py                    # Streamlit 4-page dashboard
├── src/
│   ├── features.py           # Feature engineering pipeline
│   ├── segmentation.py       # K-Means++, GMM, UMAP, bootstrap stability
│   ├── churn_model.py        # Per-segment XGBoost + calibration + MLflow
│   ├── uplift_model.py       # CausalML T-Learner + S-Learner + ROI
│   ├── retention_llm.py      # Claude retention action generator
│   └── pipeline.py           # Full pipeline orchestrator
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Feature-engineered and model output parquets
├── models/                   # Serialized models and stability artifacts
├── requirements.txt
└── README.md
```

## Resume Bullets

- Built a per-segment XGBoost churn system with isotonic probability calibration and gain-based feature importance, matching the architecture used by Salesforce Einstein for customer health scoring
- Implemented uplift modeling (T-Learner + S-Learner via Uber's CausalML) to shift optimization from churn probability to incremental retention value, identifying 710 Persuadables from 5,630 customers
- Applied bootstrap cluster stability analysis (Adjusted Rand Index across 100 resamplings) to validate that behavioral segments are data-robust, not random-seed artifacts
- Integrated an LLM retention action generator using Claude that converts feature importance and uplift scores into prioritized intervention strategies with channel, timing, and ROI reasoning
- Achieved CV AUC of 0.945–0.984 across 5 behavioral segments with isotonic-calibrated probabilities for accurate intervention ROI calculation
