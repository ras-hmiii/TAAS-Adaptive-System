# TAAS - Traceable and Adaptive AI System

TAAS is a research prototype for studying whether explainability, drift
detection, and controlled model adaptation can maintain model reliability when
real-world data distributions change.

## Research questions

1. Can SHAP explanations remain faithful and useful as input distributions shift?
2. How do ADWIN and Page-Hinkley differ in detection behavior?
3. When does a detected drift justify retraining or incremental adaptation?
4. Does adaptation recover performance without unacceptable forgetting?

## Current status

### Implemented

- Adult Income ingestion, cleaning, deterministic 70/15/15 splitting, and
  baseline Random Forest training.
- SHAP instance and batch explanations with reconstruction fidelity checks.
- Configurable mean, variance, and noise drift simulation.
- Structured ADWIN and Page-Hinkley detector results.
- Classification monitoring metrics and a conservative adaptation decision
  workflow. Drift does not automatically promote a model.
- In-memory model metadata registry, reusable experiment runner, and structured
  audit events.
- JSON persistence for experiment results and model metadata.
- Deterministic streaming adaptation runner with candidate evaluation and audit
  events.
- Optional ResNet50/ImageNet prediction and Grad-CAM heatmap/overlay utilities.
- FastAPI health/model/prediction endpoints.
- Streamlit overview, prediction, explainability, drift, adaptation,
  comparison, experiment, audit, and research-result pages backed by saved JSON
  results.

### In progress

- End-to-end incremental learning and EWC comparison experiments.
- Production image dataset training and evaluation beyond ImageNet inference.

### Planned

- More complete dashboard views for drift distributions, model comparison, and
  research results.
- Larger controlled experiments and statistical analysis of detection delay,
  false alarms, recovery, and forgetting.

## Architecture and data flow

```text
Adult data -> preprocessing -> baseline model -> prediction -> SHAP
           -> controlled drift -> ADWIN/Page-Hinkley -> monitoring
           -> adaptation decision -> candidate evaluation -> audit/version
```

The detector, adaptation, monitoring, and explainability modules do not depend
on Streamlit. This keeps the research logic testable and reusable from the API
or an experiment script.

## Repository layout

```text
src/
  ingestion/       Adult dataset loading and preprocessing
  models/          baseline model training
  explainability/  SHAP tabular explanations
  drift/           detectors and controlled drift simulation
  monitoring/      classification metrics
  adaptation/      decisions and model metadata registry
  experiments/     reproducible experiment primitives
  audit/           chronological structured events
  api/             FastAPI backend
dashboard/         Streamlit research console
tests/             behavioral tests
data/              local raw and processed datasets
```

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the system

Train the baseline (downloads Adult Income if it is not cached):

```bash
python -m src.models.train_tabular
```

Start the API:

```bash
python -m src.api
```

Available backend routes are `/health`, `/models`, `/predictions`,
`/explanations`, `/experiments`, `/results`, `/drift`, and `/adaptation`.

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

Run tests:

```bash
python -m pytest -q
```

## Metrics and research validity

Metrics are calculated from supplied observations; the dashboard does not
invent health or performance values. If no experiment JSON files exist under
`results/`, it explicitly reports that results are unavailable. The current
prototype calculates accuracy, precision, recall, F1, and ROC-AUC when a
two-class sample is available. Detection delay, false alarms, recovery time,
and forgetting require a completed streaming experiment and are therefore not
claimed as implemented results.

## Data

Data files are intentionally not committed. The Adult Income source is the UCI
repository; the loader also supports an OpenML fallback. Place local files
under `data/raw/` when offline.

## Limitations and next experiment

The current model is a tabular Random Forest and the registry is process-local.
The recommended next experiment is a deterministic Adult Income stream with a
configured mean shift at a known batch, comparing no adaptation, periodic
retraining, and incremental adaptation with both detectors. Record pre-drift,
post-drift, recovery, detection delay, and candidate-promotion decisions.
