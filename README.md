# IBM-HR-Analytics
Unsupervised K-Means segmentation of IBM's HR workforce with an interactive Gradio dashboard for attrition risk and retention scenario simulation.
# IBM HR Analytics — Employee Segmentation & Attrition Risk Explorer

Unsupervised K-Means clustering app that segments IBM's HR workforce, surfaces attrition risk by segment, and lets HR partners score hypothetical employee profiles against real workforce personas — built with Gradio + scikit-learn.

🔗 **Live demo:** [Hugging Face IBM Dashboard](https://huggingface.co/spaces/jersross/ibm-attrition)

## What it does

IBM's Chief HR Officer needed a way to understand attrition risk beyond single KPIs or manual demographic cuts. This app applies **K-Means clustering** to 1,470 employee records to discover naturally occurring workforce segments — without ever using `Attrition` as a training label — then profiles each segment's attrition rate, compensation, tenure, and satisfaction after the fact.

The model dynamically evaluates every cluster count from k=2 to k=10 at startup and defaults the dashboard to whichever k scores best on silhouette score, rather than a hardcoded value. On the current dataset that resolves to **k=7**, uncovering segments ranging from a Senior Leadership tier under 5% attrition to a High-Risk segment above 22%.

## Features

The app is organized into four tabs:

- **📊 Executive Overview** — Company-wide KPIs and auto-generated, data-grounded answers to five core HR business questions (highest-risk segment, most stable segment, where headcount concentrates, highest-paid segment, weakest satisfaction).
- **📈 Model Evaluation & Validation** — Elbow method and silhouette score curves across k=2–10, with the model's chosen k highlighted and an explanation of why.
- **🔍 Cluster Profiles & Segment Intelligence** — Interactive k selector, attrition/income/satisfaction charts by segment, and a full cluster profile table with auto-generated business labels (e.g. *High-Risk Entry-Level*, *Senior Leadership Anchor*, *Entry-Level Tenured*).
- **🧪 Scenario Simulation & Risk Classifier** — Input a hypothetical employee's profile and see which real workforce segment they'd be nearest to, that segment's historical attrition rate, and tailored retention interventions.

## How it works

- **Preprocessing:** a single scikit-learn `Pipeline` (`ColumnTransformer` + `StandardScaler`) one-hot encodes `Department`/`Gender`, passes ordinal/numeric fields through, and scales everything — reused unchanged for both training and live scenario scoring.
- **Clustering:** 10 `KMeans` models (k=2–10, `n_init=10`, fixed random state) are fit once at startup and cached, so switching k in the UI is instant.
- **Model selection:** k is chosen dynamically from whichever candidate scores highest on silhouette score — never hardcoded.
- **Segment labeling:** each cluster is labeled programmatically from its z-scored deviation from the overall employee population (age, tenure, income, job level, satisfaction) and its attrition rank relative to peer clusters — not from fixed thresholds tuned to one dataset.
- **Fallback data:** if `IBMDataset.csv` isn't found, a schema-matching synthetic dataset is generated so the app never crashes during deployment/build checks.

## Tech stack

`Python` · `Gradio` · `scikit-learn` (KMeans, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline) · `pandas` / `numpy` · `Plotly`

## Running locally

```bash
pip install gradio pandas numpy plotly scikit-learn
python app.py
```

Place `IBMDataset.csv` in the project root (the standard IBM HR Employee Attrition dataset, 1,470 rows / 13 fields — see below). If the file isn't present, the app generates a synthetic dataset with the same schema so it still runs.

## Data

- **Source:** IBM HR Employee Attrition dataset (1,470 employees, 13 features: Age, Attrition, Department, Education, Gender, JobLevel, JobSatisfaction, MonthlyIncome, TotalWorkingYears, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager).
- `Attrition` is used exclusively for post-hoc cluster profiling — it is never a clustering input, keeping the segmentation strictly unsupervised.

## Project structure

```
app.py    # Single-file Gradio app: data loading, preprocessing pipeline,
          # clustering engine, business-question generator, chart builders,
          # and the 4-tab UI
```

---

Built as a portfolio project applying unsupervised learning to HR analytics — from raw employee records to an interactive, decision-ready tool for retention strategy.
