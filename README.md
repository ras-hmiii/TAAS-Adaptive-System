# TAAS - Traceable and Adaptive AI Systems

This is my project for the AI/ML Microdegree program at AI Community in Nepal (AICN).

Supervisor: Diwas Sapkota

## What this is about

Most AI models have two problems. First, you can't really tell why they made
a decision (the "black box" thing). Second, once you deploy them, the real
world data changes over time and the model just quietly gets worse without
anyone noticing (this is called concept drift).

This project tries to deal with both issues in one pipeline - using SHAP and
GRAD-CAM for explanations, and drift detection + incremental learning so the
model can adapt instead of silently failing.

## Current progress

Right now I have the tabular pipeline working:
- loads the UCI Adult Income dataset and cleans/encodes it
- trains a Random Forest baseline model
- generates SHAP explanations for individual predictions
- basic test suite for all of the above

Still working on:
- image model (ResNet50) + GRAD-CAM
- drift detection (ADWIN, Page-Hinkley)
- incremental learning with EWC
- dashboard (Streamlit)
- REST API (FastAPI)

I'll keep updating this as I go, following the development plan from the
proposal (roughly 16 weeks, phase by phase).

## Folder structure

```
taas/
├── src/
│   ├── ingestion/       data loading and preprocessing
│   ├── models/          model training
│   ├── explainability/  SHAP and GRAD-CAM
│   ├── drift/           drift detection (coming soon)
│   ├── adaptation/      incremental learning (coming soon)
│   └── api/             FastAPI app (coming soon)
├── dashboard/           Streamlit app (coming soon)
├── notebooks/           exploratory analysis
├── tests/               pytest tests
├── data/                raw and processed data (not committed, see below)
└── docs/
```

## How to run it

```bash
git clone https://github.com/<username>/traceable-adaptive-ai-systems.git
cd traceable-adaptive-ai-systems

python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Train the baseline model:
```bash
python -m src.models.train_tabular
```

Run the tests:
```bash
pytest --cov=src
```

## Datasets

I didn't commit the actual data files (too big, plus licensing). You'll need
to grab them yourself:

- UCI Adult Income: https://archive.ics.uci.edu/dataset/2/adult
- Chest X-Ray Pneumonia (Kaggle): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Put them in `data/raw/`. The Adult Income one will actually download
automatically when you run the training script if it's not already there.

## Tech used

Python, scikit-learn, SHAP, TensorFlow (for the image part later), River
(for drift detection later), Streamlit, FastAPI, pytest

## License

MIT, see LICENSE file
