# Customer Segmentation & Churn Engine

A **decision intelligence platform** for customer retention that mirrors the architecture used by Uber, Netflix, and Salesforce in production. Not just a churn model — a full closed-loop system that segments customers behaviorally, predicts churn per cohort, identifies which customers will *respond* to intervention (uplift modeling), and deploys a 12-tool AI agent that generates, saves, and tracks personalized retention strategies.

---

## What Makes This Different

Most churn projects do: `features → model → churn probability → email everyone above 0.7`

This system does what production systems actually do:

```
raw behavioral data
  → 8 engineered composite features (engagement score, recency decay, stickiness index, spend trend)
  → K-Means++ segmentation + GMM soft probability assignments
  → bootstrap ARI stability validation (100 resamplings)
  → per-segment CatBoost classifiers (separate model per cohort)
  → isotonic probability calibration (required for ROI calculations)
  → uplift modeling (CausalML T-Learner + S-Learner)
  → intervention ROI ranking (who is actually worth spending on)
  → 12-tool ReAct AI agent (what to do, which channel, why, then saves the action)
  → audit trail + outcome tracking (closed-loop feedback)
```

---

## The Uplift Difference

A naive churn model wastes retention budget on three wrong populations:

| Customer Type | Description | Action |
|---|---|---|
| **Sure Things** | Will stay regardless of intervention | Don't spend |
| **Lost Causes** | Will churn regardless of intervention | Don't spend |
| **Sleeping Dogs** | Would stay but intervention triggers churn | Don't spend |
| **Persuadables** | High churn risk AND will respond positively | **Target these** |

Uber open-sourced CausalML for exactly this use case. Netflix uses uplift modeling for campaign targeting. This project implements both S-Learner and T-Learner approaches.

**Result on the e-commerce dataset: 710 Persuadables identified out of 948 high-risk customers** — avoiding wasted spend on 238 Lost Causes.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Python ML Pipeline                     │
│  features.py → segmentation.py → churn_model.py →       │
│  uplift_model.py → database.py (PostgreSQL)              │
└────────────────────────┬────────────────────────────────┘
                         │ Supabase (PostgreSQL)
                         │ 5 tables · 10 RPCs
┌────────────────────────▼────────────────────────────────┐
│                 Next.js 16 Dashboard                     │
│  Segmentation · Churn · Uplift · Retention · Analytics  │
│  + /api/agent — 12-tool ReAct agent (Groq)              │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | What It Does | Key Tech |
|---|---|---|---|
| Feature Engineering | `features.py` / `olist_features.py` / `cell2cell_features.py` | Builds 8 composite behavioral features from raw columns | pandas, numpy, scikit-learn |
| Segmentation | `segmentation.py` | 5 behavioral cohorts with stability validation | K-Means++, GMM, PaCMAP, ARI |
| Churn Prediction | `churn_model.py` | Per-segment binary classifiers with calibration | CatBoost, isotonic regression, MLflow |
| Uplift Modeling | `uplift_model.py` | Identifies Persuadables and ranks by ROI | CausalML, XGBoost, T-Learner, S-Learner |
| Persistence | `database.py` | Audit trail for recommendations and feedback | PostgreSQL, psycopg2 |

### Dashboard Pages

| Page | What It Shows |
|---|---|
| **Segmentation** | Segment heatmap, PaCMAP behavioral scatter, GMM soft probability distributions |
| **Churn** | KPI cards, churn probability histogram, SHAP feature importance, risk tier breakdown by segment |
| **Uplift** | Customer type funnel (Persuadable / Sure Thing / Lost Cause / Sleeping Dog), ROI by segment, top persuadable priority list |
| **Retention** | AI agent interface — batch auto-generate or conversational chat, agent trace timeline, action cards |
| **Analytics** | Audit log of all AI-generated actions, outcome feedback (retained / churned / pending), success rate by intervention type |

---

## AI Agent

The retention agent runs a **ReAct loop** (max 5 rounds) powered by `llama-3.3-70b-versatile` on Groq. It has access to 12 tools and operates in two modes:

- **Batch mode** — given a customer, auto-generates a full retention action plan and saves it to the database
- **Chat mode** — conversational assistant for the retention manager

### Tools

| # | Tool | What It Does |
|---|---|---|
| 1 | `get_top_churn_drivers` | SHAP-based churn drivers for a specific customer |
| 2 | `get_segment_benchmark` | Average metrics for a named segment |
| 3 | `calculate_intervention_roi` | Net ROI given uplift score, CLV, and cost |
| 4 | `lookup_customer_details` | Full customer record by ID |
| 5 | `search_retention_playbook` | DB-driven playbook lookup by risk factor keyword |
| 6 | `get_all_segment_benchmarks` | Cross-segment comparison in one call |
| 7 | `get_past_interventions` | Intervention history per customer (prevents repeating failed approaches) |
| 8 | `get_intervention_success_rates` | Historical retention rates by intervention type |
| 9 | `get_at_risk_customers` | Top high-risk customers, optionally by segment |
| 10 | `get_revenue_at_risk` | Expected churner count × CLV, optionally by segment |
| 11 | `save_retention_action` | Persists the recommended action to Supabase |
| 12 | `get_unactioned_persuadables` | Highest-ROI Persuadables with no action yet — the priority work queue |

The system prompt is built dynamically at request time from the `business_config` table — intervention types, channels, timing options, and assumed CLV are all DB-driven and update without a code deploy.

---

## Datasets

The pipeline supports three datasets out of the box, selectable via `--dataset`:

| Dataset | `--dataset` flag | Rows | Domain |
|---|---|---|---|
| E-Commerce Customer Churn (Kaggle) | `ecommerce` (default) | 5,630 | Retail / D2C e-commerce |
| Olist Brazilian E-Commerce (Kaggle) | `olist` | 42,325 | Multi-seller marketplace |
| Cell2Cell Telecom Churn | `cell2cell` | ~71,000 | Subscription / telecom |

The e-commerce dataset includes: `HourSpendOnApp`, `DaySinceLastOrder`, `OrderCount`, `CashbackAmount`, `SatisfactionScore`, `Complain`, `NumberOfDeviceRegistered`, `NumberOfAddress`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `Tenure`, and `Churn`.

---

## Model Results (E-Commerce Dataset)

| Segment | Customers | Churn Rate | CV AUC |
|---|---|---|---|
| At-Risk | 1,228 | 23.1% | 0.982 |
| Lapsed | 1,482 | 6.3% | 0.974 |
| Price Sensitive | 889 | 40.4% | 0.945 |
| Champions | 734 | 12.8% | 0.976 |
| Loyal Customers | 1,297 | 9.0% | 0.984 |

Cluster stability: **Mean ARI = 0.921 (Highly Stable)** across 100 bootstrap resamplings.

---

## Technology Stack

### Python Pipeline
| Library | Version | Purpose |
|---|---|---|
| pandas | 2.3.3 | Feature engineering and data wrangling |
| scikit-learn | ≥1.7.0 | Clustering, calibration, metrics |
| catboost | ≥1.2.0 | Per-segment churn classifiers |
| xgboost | 3.2.0 | Uplift model base learners |
| causalml | 0.16.0 | T-Learner and S-Learner uplift modeling |
| shap | 0.47.2 | Feature importance approximation |
| pacmap | ≥0.7.0 | 2D behavioral space visualization |
| mlflow | 3.11.1 | Per-segment experiment tracking |
| groq | ≥0.9.0 | LLM inference (Groq free tier) |
| fastapi | 0.129.0 | REST API serving layer |
| streamlit | ≥1.28.0 | Prototype dashboard |
| psycopg2-binary | ≥2.9.9 | PostgreSQL persistence |

### Next.js Dashboard
| Library | Version | Purpose |
|---|---|---|
| next | 16.2.9 | App framework (App Router) |
| react | 19.2.4 | UI |
| typescript | ^5 | Type safety |
| @supabase/supabase-js | ^2.108.2 | Database client |
| groq-sdk | ^1.3.0 | AI agent inference |
| recharts | ^3.9.0 | Bar/line/area charts |
| react-plotly.js | ^4.0.0 | Scatter plots (PaCMAP, uplift) |
| tailwindcss | ^4 | Styling |
| @base-ui/react | ^1.6.0 | Accessible UI primitives |

---

## Database Schema

### Tables

**`customers`** — enriched output of the full ML pipeline. One row per customer.

Key columns: `customer_id`, `segment`, `churn_probability`, `risk_tier`, `uplift_score`, `customer_type`, `net_roi`, `roi_positive`, `intervention_priority`, `umap_1`, `umap_2`, `top_shap_features` (JSON), `gmm_prob_seg0–4`, plus raw behavioral features.

**`retention_actions`** — audit log of every AI-generated retention recommendation.

Key columns: `id` (UUID), `customer_id`, `segment`, `churn_probability`, `uplift_score`, `net_roi`, `intervention_type`, `channel`, `timing`, `message_framing`, `confidence`, `agent_reasoning` (JSON trace), `agentic_mode`, `generated_at`.

**`intervention_feedback`** — CSM outcome feedback.

Key columns: `id`, `retention_action_id`, `customer_id`, `outcome` (retained / churned / pending).

**`retention_playbook`** — DB-driven playbook for the `search_retention_playbook` tool. Edit rows here instead of touching source code.

| Column | Description |
|---|---|
| `risk_factor_keyword` | Keyword matched against churn drivers (e.g. `satisfaction`, `complain`) |
| `intervention` | Recommended intervention type |
| `message` | Guidance for the retention manager |
| `cost` | Estimated cost band |

**`business_config`** — key-value store for runtime parameters. The AI agent reads these at request time — no code deploy needed to change them.

| Key | Default | Description |
|---|---|---|
| `assumed_clv_usd` | `500` | Customer Lifetime Value used in revenue-at-risk calculations |
| `intervention_types` | 7 types | Comma-separated list the agent can choose from |
| `channels` | 5 channels | Valid outreach channels |
| `timing_options` | 4 options | Valid timing options |

### Supabase RPCs

The dashboard uses the following PostgreSQL RPC functions:

| Function | Used By |
|---|---|
| `get_segment_summary()` | Segmentation page |
| `get_churn_kpis(p_segment)` | Churn page, revenue-at-risk tool |
| `get_churn_histogram(p_segment)` | Churn page |
| `get_risk_summary()` | Churn page |
| `get_shap_summary(p_segment)` | Churn page |
| `get_avg_churn_by_segment()` | Churn page |
| `get_customer_type_summary()` | Uplift page |
| `get_roi_by_segment()` | Uplift page |
| `get_top_persuadables(p_limit)` | Uplift page |
| `get_uplift_kpis()` | Uplift page |

---

## Project Structure

```
Customer-Segmentation-Churn-Engine/
├── src/
│   ├── pipeline.py           # Orchestrator — runs all 4 stages, smart caching
│   ├── features.py           # E-commerce feature engineering (8 composite features)
│   ├── olist_features.py     # Olist (Brazilian marketplace) feature engineering
│   ├── cell2cell_features.py # Cell2Cell telecom feature engineering
│   ├── segmentation.py       # K-Means++, GMM, PaCMAP, bootstrap ARI stability
│   ├── churn_model.py        # Per-segment CatBoost + isotonic calibration + MLflow
│   ├── uplift_model.py       # T-Learner + S-Learner (CausalML) + ROI ranking
│   ├── retention_llm.py      # Groq-backed retention action generator
│   ├── agent_loop.py         # ReAct agent loop implementation
│   ├── agent_tools.py        # Tool implementations for the agent
│   ├── database.py           # PostgreSQL persistence layer (graceful degradation)
│   └── logging_config.py     # Structured logging configuration
│
├── dashboard/                # Next.js 16 production dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── segmentation/page.tsx
│   │   │   ├── churn/page.tsx
│   │   │   ├── uplift/page.tsx
│   │   │   ├── retention/page.tsx
│   │   │   ├── analytics/page.tsx
│   │   │   └── api/agent/route.ts  # 12-tool ReAct agent API
│   │   ├── components/
│   │   │   ├── pages/              # Client components (charts, agent UI, audit)
│   │   │   └── ui/                 # Shared UI primitives
│   │   └── lib/
│   │       ├── data.ts             # Supabase RPC wrappers and typed queries
│   │       └── supabase.ts         # Client init + TypeScript types
│   └── next.config.ts              # Loads root .env via dotenv at build/dev time
│
├── supabase/
│   └── config_tables.sql     # DDL for retention_playbook and business_config tables
│
├── data/
│   └── processed/            # Pipeline outputs (parquet files — tracked by git)
│
├── models/                   # Serialized models and stability artifacts (tracked by git)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Supabase](https://supabase.com) project (free tier works)
- A [Groq](https://console.groq.com) API key (free tier works)
- Kaggle credentials (only for downloading the raw dataset)

### 1. Clone and configure

```bash
git clone https://github.com/shiva-shivanibokka/Customer-Segmentation-Churn-Engine.git
cd Customer-Segmentation-Churn-Engine

cp .env.example .env
# Fill in your values in .env — this single file is read by both the
# Python pipeline and the Next.js dashboard
```

### 2. Run the ML pipeline

```bash
pip install -r requirements.txt

# Download the default e-commerce dataset
kaggle datasets download -d ankitverma2010/ecommerce-customer-churn-analysis-and-prediction \
    -p data/raw --unzip

# Run the full 4-stage pipeline (segments, churn models, uplift, ROI)
python src/pipeline.py

# Or force a full retrain:
python src/pipeline.py --force

# Or run on a different dataset:
python src/pipeline.py --dataset olist
python src/pipeline.py --dataset cell2cell
```

Artifacts are cached to `data/processed/` and `models/`. Re-running without `--force` loads from cache.

### 3. Load your Supabase tables

In your Supabase SQL editor, run the contents of `supabase/config_tables.sql` to create the `retention_playbook` and `business_config` tables.

The `customers`, `retention_actions`, and `intervention_feedback` tables are created by the pipeline's database layer (`src/database.py`) on first run, or you can create them manually to match the schema in `dashboard/src/lib/supabase.ts`.

### 4. Launch the dashboard

```bash
cd dashboard
npm install
npm run dev
```

The dashboard starts on `http://localhost:3000`. It reads all environment variables from the root `.env` file automatically via `next.config.ts`.

---

## Environment Variables

A single `.env` file at the repo root is used by both the Python pipeline and the Next.js dashboard.

```bash
# Supabase — all three values from: Project → Settings → API
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...          # Browser-safe reads
SUPABASE_SERVICE_ROLE_KEY=eyJ...              # Server-side writes (bypasses RLS)

# Direct PostgreSQL connection — Project → Settings → Database → URI
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# Groq (free tier at console.groq.com)
GROQ_API_KEY=gsk_...

# Kaggle (only needed for dataset download)
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-kaggle-api-key
```

> **Why two Supabase keys?** The anon key is used for all reads (respects Row Level Security). The service role key is used only in the server-side API route for inserting retention actions — this bypasses RLS and is never exposed to the browser.

---

## Industry Parallels

| This Project | Production System |
|---|---|
| Per-segment CatBoost with isotonic calibration | Salesforce Einstein per-tier customer health scoring |
| CausalML T-Learner + S-Learner uplift | Uber's production retention campaign targeting |
| Bootstrap ARI cluster stability (100 resamplings) | Production ML segment validation |
| GMM soft probability assignments | Boundary-handling for ambiguous health scores |
| 12-tool ReAct AI agent for retention playbooks | Salesforce Einstein Copilot CSM recommendations |
| PaCMAP behavioral space visualization | Netflix member segment exploration |
| DB-driven system prompt (business_config) | Production LLM config without code deploys |

---

## Design Decisions

**Per-segment models over a single global model.** A Champion churns for different reasons than a Lapsed customer. Champions who churn usually have a specific trigger (bad support experience, competitor offer). Lapsed customers churn through gradual disengagement. Separate models capture segment-specific dynamics.

**Isotonic calibration over raw probabilities.** Raw CatBoost probabilities are not well-calibrated — a score of 0.7 does not mean 70% of customers at that score actually churn. Calibration is required whenever probabilities drive business calculations (CLV, retention ROI, budget allocation).

**SHAP approximation over TreeExplainer.** CatBoost/XGBoost 3.x changes to `base_score` handling introduce instability in interaction value computation. This project uses deviation-weighted feature importance (global gain scores × individual feature deviation from segment mean). Fast, stable, and sufficient for ranking churn drivers.

**Observational uplift modeling.** The e-commerce dataset has no historical A/B test. The project uses a documented simulation strategy: `Complain` flag = proxy for received support outreach (treated); `CouponUsed > 0` = proxy for received discount offers (treated). This matches academic literature on observational uplift. Production systems (Uber, Netflix) train on actual randomized experiment logs.

**Server-side retention action saves.** The `retention_actions` table has Row Level Security enabled. The AI agent API route uses the service role key (server-side only) for inserts, so the anon key never needs write permissions. All client-side code is read-only.

**Dynamic agent configuration.** The system prompt is rebuilt from the `business_config` table on every request. Changing CLV assumptions, intervention types, or channels requires only a database row update — no code change, no deploy.

---

## Resume Bullets

- Built a per-segment CatBoost churn system with isotonic probability calibration and gain-based feature importance, matching Salesforce Einstein's per-tier customer health scoring architecture
- Implemented T-Learner and S-Learner uplift modeling (Uber's CausalML) to shift optimization from churn probability to incremental retention value — identified 710 Persuadables from 5,630 customers, avoiding wasted spend on Lost Causes and Sleeping Dogs
- Applied bootstrap cluster stability validation (Adjusted Rand Index across 100 resamplings, mean ARI = 0.921) to confirm behavioral segments are data-robust, not random-seed artifacts
- Deployed a 12-tool ReAct AI agent (Groq llama-3.3-70b-versatile) that reasons over SHAP drivers, segment benchmarks, intervention history, and ROI calculations before generating and persisting a personalized retention plan
- Built a Next.js 16 / React 19 analytics dashboard with 5 pages, 10 Supabase RPCs, Recharts/Plotly visualizations, and a closed-loop feedback system (CSM marks outcomes as retained / churned / pending)
- Achieved CV AUC of 0.945–0.984 across 5 behavioral segments; pipeline supports three datasets (e-commerce, Olist, Cell2Cell) selectable via CLI flag
