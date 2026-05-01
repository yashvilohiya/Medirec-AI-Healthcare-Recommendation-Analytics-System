# MediRec 🩺

A Flask-based medical data analytics dashboard that combines symptom-based diagnosis with statistical analysis and data science experiments.

## Features

- **Symptom Diagnosis** — Match patient symptoms to diseases using cosine similarity, fuzzy matching, or a hybrid algorithm
- **8 Analytics Experiments:**
  | # | Experiment | Technique |
  |---|-----------|-----------|
  | 1 | Age Distribution | Box Plot + Histogram |
  | 2 | Age vs Symptom Count | Linear Regression |
  | 3 | Population Sampling | Random & Systematic |
  | 4 | Patient Clustering | K-Means |
  | 5 | Disease Probability | Normal / Binomial / Poisson |
  | 6 | Statistical Properties | Mean, Median, Skewness, Kurtosis |
  | 7 | Hypothesis Testing | T-test, Chi-Square, Z-scores |
  | 8 | Trend Forecasting | Time Series + Moving Averages |

## Tech Stack

`Python` `Flask` `Pandas` `NumPy` `SciPy` `scikit-learn`

## Getting Started

```bash
git clone https://github.com/your-username/medirec.git
cd medirec
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project Structure

```
medirec/
├── app.py              # Flask backend & all API routes
├── index.html          # Frontend dashboard
├── medical_data.csv    # Patient dataset
└── requirements.txt
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/diagnose` | Symptom-based disease prediction |
| `GET /api/exp1–8` | Statistical experiment data |
| `GET /api/summary` | Dataset overview |

## Dataset

~289 patient records with fields: `Name`, `DateOfBirth`, `Gender`, `Symptoms`, `Causes`, `Disease`, `Medicine`.
