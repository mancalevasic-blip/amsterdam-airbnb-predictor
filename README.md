# Amsterdam Airbnb Stay Value Predictor

**MADA Capstone — Manca Levašič · June 2026**

Predict the **nightly price** and **booking demand** of Amsterdam Airbnb listings using gradient-boosted machine learning models trained on 10,480 listings from [Inside Airbnb](https://insideairbnb.com/).

---

## Live App

**[Open the Streamlit App →](https://amsterdam-airbnb-predictor.streamlit.app)**

---

## Project Structure

```
.
├── data/raw/listings.csv        # Inside Airbnb Amsterdam (Sept 2025)
├── models/
│   ├── price_model.pkl          # XGBoost regression pipeline (log-price)
│   ├── avail_model.pkl          # XGBoost classification pipeline (low_availability)
│   └── metadata.json            # Feature lists and metadata
├── app/app.py                   # Streamlit prediction app
├── slides/slides.qmd            # Quarto reveal.js presentation
├── ml-analysis.qmd              # Full reproducible analysis report
├── executive_summary.md         # One-page plain-language summary
├── ai_workflow_reflection.md    # AI tools / verification reflection
└── requirements.txt
```

## Deliverables

| # | Deliverable | File |
|---|-------------|------|
| D1 | Reproducible analysis report | `ml-analysis.qmd` |
| D2 | Deployed web app | `app/app.py` → Streamlit Cloud |
| D3 | AI workflow reflection | `ai_workflow_reflection.md` |
| D4 | Presentation slides | `slides/slides.qmd` |
| D5 | Executive summary | `executive_summary.md` |

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires `libomp`. If you get an import error, run:
> ```bash
> brew install libomp
> ```
> Or the app will auto-detect and use the libomp bundled with scikit-learn.

### 2. Download the data

The raw data file is not committed to the repo (too large). Download it:

```bash
mkdir -p data/raw
curl -L -o data/raw/listings.csv.gz \
  "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2025-09-11/data/listings.csv.gz"
gunzip data/raw/listings.csv.gz
```

### 3. Run the analysis report

```bash
quarto render ml-analysis.qmd
```

This trains the models and saves them to `models/`. The rendered HTML report will appear as `ml-analysis.html`.

### 4. Run the Streamlit app locally

```bash
streamlit run app/app.py
```

### 5. Render the slides

```bash
quarto render slides/slides.qmd
```

---

## Model Performance (test set, 15% held-out)

| Task | Model | Metric | Baseline | XGBoost |
|------|-------|--------|----------|---------|
| Price regression | XGBoost | R² | 0.00 | **0.63** |
| Price regression | XGBoost | MAE | €100 | **€63** |
| Availability classification | XGBoost | ROC-AUC | 0.50 | **0.70** |

---

## Data

**Source:** [Inside Airbnb](https://insideairbnb.com/) — Amsterdam, North Holland, Netherlands  
**Snapshot:** September 11, 2025  
**License:** Creative Commons Attribution 4.0  
**Rows:** 10,480 · **Columns:** 79
