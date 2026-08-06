"""
IBM HR Analytics — Employee Segmentation & Attrition Risk Explorer
====================================================================
Unsupervised K-Means clustering application built for IBM CHRO Dr. Sarah Chen's
strategic initiative to discover employee segments, analyze attrition
characteristics, evaluate clustering quality, and simulate risk segments for
hypothetical employee profiles.

Self-contained Gradio app intended for deployment on Hugging Face Spaces.
Reads IBMDataset.csv from the working directory; if the file is missing, a
schema-matching synthetic dataset is generated so the app never crashes
during build/deployment checks.

Run locally:
    pip install gradio pandas numpy plotly scikit-learn
    python app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "IBMDataset.csv"
# Accept common filename variants so the app finds the dataset regardless of
# minor naming differences between the spec and the uploaded file.
DATA_PATH_CANDIDATES = ["IBMDataset.csv", "IBM_Dataset.csv", "IBM-Dataset.csv"]
RANDOM_STATE = 42
K_RANGE = list(range(2, 11))          # k = 2 .. 10
# NOTE: there is no hardcoded default k. The app's default is computed at
# startup as the silhouette-optimal k (see OPTIMAL_K below, set once the
# engine has fit and scored every k from 2-10) and reused everywhere a
# default is needed.

NOMINAL_COLS = ["Department", "Gender"]
ORDINAL_NUMERIC_COLS = [
    "Education", "JobLevel", "JobSatisfaction", "WorkLifeBalance",
    "Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
    "YearsInCurrentRole", "YearsWithCurrManager",
]
CLUSTER_FEATURE_COLS = NOMINAL_COLS + ORDINAL_NUMERIC_COLS

DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
GENDERS = ["Female", "Male"]


# ---------------------------------------------------------------------------
# Data loading (real file, with synthetic fallback so the app never crashes)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n=1470, seed=RANDOM_STATE):
    """Generates a schema-matching synthetic dataset if IBMDataset.csv is absent."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 61, n)
    department = rng.choice(DEPARTMENTS, n, p=[0.12, 0.55, 0.33])
    education = rng.integers(1, 6, n)
    gender = rng.choice(GENDERS, n, p=[0.4, 0.6])
    job_level = rng.integers(1, 6, n)
    job_satisfaction = rng.integers(1, 5, n)
    monthly_income = rng.integers(1009, 20000, n)
    total_working_years = np.clip(
        (age - 18 - rng.integers(0, 6, n)).astype(int), 0, 40
    )
    work_life_balance = rng.integers(1, 5, n)
    years_at_company = np.clip(
        (total_working_years - rng.integers(0, 5, n)).astype(int), 0, 40
    )
    years_in_current_role = np.clip(
        (years_at_company - rng.integers(0, 4, n)).astype(int), 0, 18
    )
    years_with_curr_manager = np.clip(
        (years_at_company - rng.integers(0, 4, n)).astype(int), 0, 17
    )

    # Attrition probability skewed by known real-world risk drivers so the
    # synthetic data still produces meaningful clusters/segments.
    risk_score = (
        (job_satisfaction <= 2).astype(float) * 0.25
        + (work_life_balance <= 2).astype(float) * 0.20
        + (monthly_income < 3500).astype(float) * 0.20
        + (years_at_company < 2).astype(float) * 0.20
        + (job_level == 1).astype(float) * 0.15
    )
    attrition_prob = np.clip(0.08 + risk_score, 0.02, 0.85)
    attrition = np.where(rng.random(n) < attrition_prob, "Yes", "No")

    df = pd.DataFrame({
        "Age": age,
        "Attrition": attrition,
        "Department": department,
        "Education": education,
        "Gender": gender,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "TotalWorkingYears": total_working_years,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsWithCurrManager": years_with_curr_manager,
    })
    return df


def load_dataset():
    found_path = next((p for p in DATA_PATH_CANDIDATES if os.path.exists(p)), None)
    if found_path is not None:
        df = pd.read_csv(found_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        # Guard against a real file that is missing expected columns.
        required = {"Attrition"} | set(CLUSTER_FEATURE_COLS)
        if not required.issubset(set(df.columns)):
            df = generate_synthetic_dataset()
    else:
        df = generate_synthetic_dataset()

    df = df.copy()
    df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core analytics engine
# ---------------------------------------------------------------------------

class HRClusteringEngine:
    """
    Fits, stores, and serves K-Means models across k = 2..10 for the IBM HR
    dataset. Preprocessing (one-hot encode nominal features, pass through
    ordinal/numeric features, scale everything) is handled by a single
    reusable scikit-learn Pipeline so training and simulation use identical
    transforms.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.preprocessor = None
        self.models = {}          # k -> fitted KMeans
        self.inertias = {}        # k -> inertia
        self.silhouette_scores = {}  # k -> silhouette score
        self.transformed_features = None
        self.feature_names = None
        self._build_preprocessor()
        self._fit_all_k()
        self._compute_population_stats()

    # -- population baseline (used for relative, data-driven labeling) ------
    def _compute_population_stats(self):
        stat_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "JobLevel",
                     "JobSatisfaction", "WorkLifeBalance"]
        self.population_means = self.df[stat_cols].mean()
        # Avoid divide-by-zero on degenerate/synthetic columns.
        self.population_stds = self.df[stat_cols].std().replace(0, 1e-9)
        self.population_attrition_rate = self.df["AttritionFlag"].mean() * 100

    # -- preprocessing -----------------------------------------------------
    def _build_preprocessor(self):
        column_transformer = ColumnTransformer(
            transformers=[
                ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS),
                ("passthrough", "passthrough", ORDINAL_NUMERIC_COLS),
            ]
        )
        self.preprocessor = Pipeline(steps=[
            ("columns", column_transformer),
            ("scale", StandardScaler()),
        ])
        X = self.df[CLUSTER_FEATURE_COLS]
        self.transformed_features = self.preprocessor.fit_transform(X)

        ohe = self.preprocessor.named_steps["columns"].named_transformers_["nominal"]
        ohe_names = list(ohe.get_feature_names_out(NOMINAL_COLS))
        self.feature_names = ohe_names + ORDINAL_NUMERIC_COLS

    # -- model fitting -------------------------------------------------------
    def _fit_all_k(self):
        for k in K_RANGE:
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = model.fit_predict(self.transformed_features)
            self.models[k] = model
            self.inertias[k] = model.inertia_
            # Silhouette score is expensive on large k / n; sample if needed.
            sample_size = min(len(labels), 1470)
            self.silhouette_scores[k] = silhouette_score(
                self.transformed_features, labels,
                sample_size=sample_size if sample_size < len(labels) else None,
                random_state=RANDOM_STATE,
            )

    def optimal_k_by_silhouette(self):
        return max(self.silhouette_scores, key=self.silhouette_scores.get)

    # -- cluster profiling ---------------------------------------------------
    def labels_for_k(self, k):
        return self.models[k].labels_

    def profile_for_k(self, k):
        """Returns a DataFrame with one row per cluster summarizing business metrics."""
        df = self.df.copy()
        df["Cluster"] = self.labels_for_k(k)

        rows = []
        for cluster_id in sorted(df["Cluster"].unique()):
            sub = df[df["Cluster"] == cluster_id]
            top_dept = sub["Department"].mode().iat[0] if not sub.empty else "N/A"
            row = {
                "Cluster": int(cluster_id),
                "Size": len(sub),
                "AvgAge": sub["Age"].mean(),
                "AvgMonthlyIncome": sub["MonthlyIncome"].mean(),
                "AttritionRatePct": sub["AttritionFlag"].mean() * 100,
                "PctFemale": (sub["Gender"] == "Female").mean() * 100,
                "TopDepartment": top_dept,
                "AvgJobSatisfaction": sub["JobSatisfaction"].mean(),
                "AvgWorkLifeBalance": sub["WorkLifeBalance"].mean(),
                "AvgYearsAtCompany": sub["YearsAtCompany"].mean(),
                "AvgYearsInCurrentRole": sub["YearsInCurrentRole"].mean(),
                "AvgTotalWorkingYears": sub["TotalWorkingYears"].mean(),
                "AvgJobLevel": sub["JobLevel"].mean(),
            }
            rows.append(row)
        profile = pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
        profile["SegmentLabel"] = self._assign_labels(profile)
        return profile

    def _assign_labels(self, profile: pd.DataFrame):
        """
        Assigns human-readable business labels using each cluster's feature
        signature *relative to the overall employee population* (z-scores)
        and *relative to its peer clusters' attrition rates* for this k —
        rather than fixed absolute cutoffs. This keeps labeling adaptive to
        whatever dataset/k is actually loaded, instead of hard-coding
        business thresholds that only fit one dataset.
        """
        attrition_median = profile["AttritionRatePct"].median()
        attrition_max = profile["AttritionRatePct"].max()

        labels = [
            self._label_segment(row, attrition_median, attrition_max)
            for _, row in profile.iterrows()
        ]

        # Disambiguate duplicate labels (common when several clusters share a
        # similar life-stage signature) by tagging with their cluster id, so
        # every dropdown/table entry stays uniquely identifiable.
        counts = pd.Series(labels).value_counts()
        dupes = set(counts[counts > 1].index)
        if dupes:
            labels = [
                f"{lbl} (Cluster {cid})" if lbl in dupes else lbl
                for lbl, cid in zip(labels, profile["Cluster"])
            ]
        return labels

    def _label_segment(self, row, attrition_median, attrition_max):
        """Labels a single cluster row based on z-scored deviation from the
        overall employee population plus its attrition rate relative to
        peer clusters within the same k."""
        pm, ps = self.population_means, self.population_stds

        age_z = (row["AvgAge"] - pm["Age"]) / ps["Age"]
        tenure_z = (row["AvgYearsAtCompany"] - pm["YearsAtCompany"]) / ps["YearsAtCompany"]
        income_z = (row["AvgMonthlyIncome"] - pm["MonthlyIncome"]) / ps["MonthlyIncome"]
        joblevel_z = (row["AvgJobLevel"] - pm["JobLevel"]) / ps["JobLevel"]
        satisfaction_z = (row["AvgJobSatisfaction"] - pm["JobSatisfaction"]) / ps["JobSatisfaction"]
        wlb_z = (row["AvgWorkLifeBalance"] - pm["WorkLifeBalance"]) / ps["WorkLifeBalance"]
        attrition_vs_company = row["AttritionRatePct"] - self.population_attrition_rate

        # Senior leadership: meaningfully above population on both seniority
        # and pay — a relative comparison, not a fixed dollar/level cutoff.
        if joblevel_z > 0.75 and income_z > 0.75:
            return "Senior Leadership Anchor"

        is_entry_level = joblevel_z < -0.5

        # High risk: this cluster's attrition sits at or near the top among
        # its peer clusters for this k, AND is above the company-wide rate.
        is_top_attrition_cluster = (
            row["AttritionRatePct"] >= attrition_max - 1e-9
            or row["AttritionRatePct"] >= attrition_median * 1.15
        )
        if attrition_vs_company > 0 and is_top_attrition_cluster:
            if is_entry_level:
                return "High-Risk Entry-Level"
            if age_z < -0.25 or tenure_z < -0.25:
                return "High-Risk Early Career"
            if satisfaction_z < -0.25 or wlb_z < -0.25:
                return "High-Risk Disengaged"
            return "High-Risk Segment"

        # Job level is otherwise the most defining trait of this cluster —
        # call it out explicitly rather than falling through to a generic
        # life-stage label that hides seniority.
        if is_entry_level:
            if tenure_z > 0.5:
                return "Entry-Level Tenured"
            return "Entry-Level Workforce"

        if tenure_z > 0.75:
            return "Mid-Career Tenured"
        if age_z < -0.5 and tenure_z < -0.5:
            return "Early Career Growth"
        if satisfaction_z > 0.25 and wlb_z > 0.25 and attrition_vs_company < 0:
            return "Stable & Engaged Core"
        return "Established Mid-Career"

    # -- scenario simulation ---------------------------------------------------
    def predict_cluster(self, k, employee_dict):
        """Transforms a single hypothetical employee profile and assigns it
        to the nearest cluster centroid for the chosen k."""
        row = pd.DataFrame([employee_dict])[CLUSTER_FEATURE_COLS]
        X = self.preprocessor.transform(row)
        cluster_id = int(self.models[k].predict(X)[0])
        return cluster_id


# ---------------------------------------------------------------------------
# Load data & fit engine once at startup
# ---------------------------------------------------------------------------

RAW_DF = load_dataset()
ENGINE = HRClusteringEngine(RAW_DF)

TOTAL_EMPLOYEES = len(RAW_DF)
OVERALL_ATTRITION_RATE = RAW_DF["AttritionFlag"].mean() * 100
SILHOUETTE_OPTIMAL_K = ENGINE.optimal_k_by_silhouette()
# The single source of truth for "the default k" used across all tabs —
# computed from the silhouette scores just fit above, never hardcoded.
OPTIMAL_K = SILHOUETTE_OPTIMAL_K


# ---------------------------------------------------------------------------
# Retention interventions — mapped to segment labels (business logic layer)
# ---------------------------------------------------------------------------

INTERVENTION_LIBRARY = {
    "High-Risk Early Career": [
        "Assign a formal mentor and structured 90-day onboarding check-ins.",
        "Introduce accelerated development / rotation programs to signal growth path.",
        "Review starting compensation against market bands for this role and department.",
    ],
    "High-Risk Disengaged": [
        "Schedule 1:1 stay interviews focused on job satisfaction and workload.",
        "Audit manager relationship quality and current work-life balance policies.",
        "Offer flexible scheduling or workload rebalancing where feasible.",
    ],
    "High-Risk Segment": [
        "Conduct targeted stay interviews to isolate the primary driver (pay, workload, or growth).",
        "Benchmark compensation and role scope against similar tenure/level peers.",
        "Flag managers of this segment for a retention-focused check-in cadence.",
    ],
    "High-Risk Entry-Level": [
        "Prioritize this segment for compensation and title-banding review — entry-level pay is a common attrition driver.",
        "Build a clear first-18-months promotion path with visible milestones.",
        "Pair every new hire with a peer mentor and structured manager check-ins.",
    ],
    "Senior Leadership Anchor": [
        "Maintain competitive total-rewards benchmarking (equity/bonus review).",
        "Provide succession-planning visibility and executive coaching investment.",
        "Protect against burnout with proactive workload and sabbatical policies.",
    ],
    "Mid-Career Tenured": [
        "Offer lateral mobility or skill-refresh programs to prevent stagnation.",
        "Recognize tenure with retention bonuses or milestone recognition.",
        "Revisit career-pathing conversations to re-engage long-tenured staff.",
    ],
    "Early Career Growth": [
        "Invest in learning & development budgets and certification support.",
        "Set clear promotion criteria and timelines to sustain momentum.",
        "Pair with senior mentors to build organizational connection.",
    ],
    "Entry-Level Workforce": [
        "Establish transparent promotion criteria from entry level to the next band.",
        "Offer skill-building programs tied to a visible growth path.",
        "Benchmark entry-level pay against market to pre-empt attrition risk.",
    ],
    "Entry-Level Tenured": [
        "Evaluate why tenure hasn't translated into job-level advancement.",
        "Offer a structured promotion or role-broadening conversation.",
        "Recognize loyalty with skills investment or a lateral growth track.",
    ],
    "Stable & Engaged Core": [
        "Sustain engagement with recognition programs; low intervention urgency.",
        "Use this group for internal referral and culture-ambassador programs.",
        "Monitor periodically to catch any emerging satisfaction decline early.",
    ],
    "Established Mid-Career": [
        "Check in on job satisfaction and work-life balance trends periodically.",
        "Offer targeted upskilling aligned to evolving role requirements.",
        "Ensure compensation remains aligned with tenure and market rates.",
    ],
}


def get_interventions(segment_label):
    # Labels may be disambiguated with a "(Cluster N)" suffix when two
    # clusters share the same base descriptor — strip it for lookup.
    base_label = segment_label.split(" (Cluster")[0]
    return INTERVENTION_LIBRARY.get(
        base_label,
        ["Conduct a stay interview to identify segment-specific retention drivers."],
    )


# ---------------------------------------------------------------------------
# Business-question narrative generator (Tab 1) — fully data-driven
# ---------------------------------------------------------------------------

def _base_label(segment_label):
    """Strips any '(Cluster N)' disambiguation suffix for display contexts
    that already show the cluster number separately."""
    return segment_label.split(" (Cluster")[0]


def build_business_questions_markdown(k):
    profile = ENGINE.profile_for_k(k)
    highest_risk = profile.loc[profile["AttritionRatePct"].idxmax()]
    lowest_risk = profile.loc[profile["AttritionRatePct"].idxmin()]
    largest = profile.loc[profile["Size"].idxmax()]
    highest_income = profile.loc[profile["AvgMonthlyIncome"].idxmax()]
    lowest_satisfaction = profile.loc[profile["AvgJobSatisfaction"].idxmin()]

    md = f"""
### 1. Which employee segments carry the highest attrition risk?
**{_base_label(highest_risk['SegmentLabel'])}** (Cluster {highest_risk['Cluster']}) shows the highest attrition
rate at **{highest_risk['AttritionRatePct']:.1f}%**, averaging age {highest_risk['AvgAge']:.0f},
tenure {highest_risk['AvgYearsAtCompany']:.1f} years, and job satisfaction
{highest_risk['AvgJobSatisfaction']:.1f}/4.

### 2. Which segment is the most stable?
**{_base_label(lowest_risk['SegmentLabel'])}** (Cluster {lowest_risk['Cluster']}) has the lowest attrition rate
at **{lowest_risk['AttritionRatePct']:.1f}%**, suggesting strong retention drivers worth
replicating elsewhere.

### 3. Where is the workforce concentrated?
The largest segment is **{_base_label(largest['SegmentLabel'])}** (Cluster {largest['Cluster']}), containing
**{largest['Size']}** employees (**{largest['Size'] / TOTAL_EMPLOYEES * 100:.1f}%** of the workforce).

### 4. Which segment commands the highest compensation, and is it at risk?
**{_base_label(highest_income['SegmentLabel'])}** (Cluster {highest_income['Cluster']}) has the highest average
monthly income at **${highest_income['AvgMonthlyIncome']:,.0f}**, with an attrition rate of
**{highest_income['AttritionRatePct']:.1f}%**.

### 5. Where is employee satisfaction weakest, and does it align with attrition?
**{_base_label(lowest_satisfaction['SegmentLabel'])}** (Cluster {lowest_satisfaction['Cluster']}) has the lowest
average job satisfaction (**{lowest_satisfaction['AvgJobSatisfaction']:.1f}/4**) and an attrition
rate of **{lowest_satisfaction['AttritionRatePct']:.1f}%**, {"reinforcing the link between low satisfaction and turnover." if lowest_satisfaction['AttritionRatePct'] > OVERALL_ATTRITION_RATE else "though attrition here remains near or below the company average, suggesting other retention factors are compensating."}
"""
    return md


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def build_elbow_chart():
    ks = K_RANGE
    inertias = [ENGINE.inertias[k] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia",
                              line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(
        x=[OPTIMAL_K], y=[ENGINE.inertias[OPTIMAL_K]],
        mode="markers+text", name=f"Optimal k={OPTIMAL_K}",
        marker=dict(color="crimson", size=14, symbol="star"),
        text=[f"k={OPTIMAL_K}"], textposition="top center",
    ))
    fig.update_layout(
        title="Elbow Method — Inertia vs. k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        template="plotly_white",
    )
    return fig


def build_silhouette_chart():
    ks = K_RANGE
    scores = [ENGINE.silhouette_scores[k] for k in ks]
    best_k = ENGINE.optimal_k_by_silhouette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=scores, mode="lines+markers", name="Silhouette Score",
                              line=dict(color="#2ca02c", width=3)))
    fig.add_trace(go.Scatter(
        x=[best_k], y=[ENGINE.silhouette_scores[best_k]],
        mode="markers+text", name=f"Peak k={best_k}",
        marker=dict(color="crimson", size=14, symbol="star"),
        text=[f"k={best_k}"], textposition="top center",
    ))
    fig.update_layout(
        title="Silhouette Score vs. k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Average Silhouette Score",
        template="plotly_white",
    )
    return fig


def build_attrition_bar_chart(profile):
    fig = px.bar(
        profile, x="SegmentLabel", y="AttritionRatePct", color="SegmentLabel",
        text=profile["AttritionRatePct"].round(1).astype(str) + "%",
        labels={"SegmentLabel": "Segment", "AttritionRatePct": "Attrition Rate (%)"},
        title="Attrition Rate by Cluster Segment",
    )
    fig.add_hline(y=OVERALL_ATTRITION_RATE, line_dash="dash", line_color="gray",
                  annotation_text=f"Company Avg: {OVERALL_ATTRITION_RATE:.1f}%")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def build_segment_comparison_chart(profile):
    metrics = ["AvgJobSatisfaction", "AvgWorkLifeBalance"]
    melted = profile.melt(
        id_vars=["SegmentLabel"], value_vars=metrics,
        var_name="Metric", value_name="Score",
    )
    label_map = {"AvgJobSatisfaction": "Job Satisfaction (1-4)",
                 "AvgWorkLifeBalance": "Work-Life Balance (1-4)"}
    melted["Metric"] = melted["Metric"].map(label_map)
    fig = px.bar(
        melted, x="SegmentLabel", y="Score", color="Metric", barmode="group",
        title="Job Satisfaction & Work-Life Balance by Segment",
        labels={"SegmentLabel": "Segment"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_income_chart(profile):
    fig = px.bar(
        profile, x="SegmentLabel", y="AvgMonthlyIncome", color="SegmentLabel",
        text=profile["AvgMonthlyIncome"].round(0).map(lambda v: f"${v:,.0f}"),
        title="Average Monthly Income by Segment",
        labels={"SegmentLabel": "Segment", "AvgMonthlyIncome": "Avg Monthly Income ($)"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def refresh_tab1(k):
    k = int(k)
    profile = ENGINE.profile_for_k(k)
    highest_risk_rate = profile["AttritionRatePct"].max()

    kpi_md = f"""
| Total Employees | Overall Attrition Rate | Optimal Clusters (k) | Highest-Risk Segment Attrition |
|:---:|:---:|:---:|:---:|
| **{TOTAL_EMPLOYEES:,}** | **{OVERALL_ATTRITION_RATE:.1f}%** | **{SILHOUETTE_OPTIMAL_K}** (silhouette-optimal) | **{highest_risk_rate:.1f}%** |
"""
    questions_md = build_business_questions_markdown(k)
    return kpi_md, questions_md


def refresh_tab3(k):
    k = int(k)
    profile = ENGINE.profile_for_k(k)
    display_cols = [
        "Cluster", "SegmentLabel", "Size", "AvgAge", "AvgMonthlyIncome",
        "AttritionRatePct", "PctFemale", "TopDepartment", "AvgJobSatisfaction",
        "AvgWorkLifeBalance", "AvgYearsAtCompany", "AvgJobLevel",
    ]
    table = profile[display_cols].round(2)
    table.columns = [
        "Cluster", "Segment Label", "Size", "Avg Age", "Avg Monthly Income",
        "Attrition Rate (%)", "% Female", "Top Department", "Avg Job Satisfaction",
        "Avg Work-Life Balance", "Avg Years at Company", "Avg Job Level",
    ]
    attrition_fig = build_attrition_bar_chart(profile)
    comparison_fig = build_segment_comparison_chart(profile)
    income_fig = build_income_chart(profile)
    return table, attrition_fig, comparison_fig, income_fig


def simulate_employee(
    k, age, department, gender, education, job_level, monthly_income,
    job_satisfaction, work_life_balance, total_working_years,
    years_at_company, years_in_current_role, years_with_curr_manager,
):
    k = int(k)
    employee = {
        "Age": age,
        "Department": department,
        "Education": education,
        "Gender": gender,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "TotalWorkingYears": total_working_years,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsWithCurrManager": years_with_curr_manager,
    }
    cluster_id = ENGINE.predict_cluster(k, employee)
    profile = ENGINE.profile_for_k(k)
    cluster_row = profile[profile["Cluster"] == cluster_id].iloc[0]

    result_md = f"""
## 🎯 Assigned Segment: **{_base_label(cluster_row['SegmentLabel'])}** (Cluster {cluster_id})

**Historical Attrition Rate for this segment: {cluster_row['AttritionRatePct']:.1f}%**
(Company average: {OVERALL_ATTRITION_RATE:.1f}%)

| Attribute | Simulated Employee | Cluster {cluster_id} Average |
|---|---:|---:|
| Age | {age} | {cluster_row['AvgAge']:.1f} |
| Monthly Income | ${monthly_income:,.0f} | ${cluster_row['AvgMonthlyIncome']:,.0f} |
| Job Satisfaction (1-4) | {job_satisfaction} | {cluster_row['AvgJobSatisfaction']:.1f} |
| Work-Life Balance (1-4) | {work_life_balance} | {cluster_row['AvgWorkLifeBalance']:.1f} |
| Years at Company | {years_at_company} | {cluster_row['AvgYearsAtCompany']:.1f} |
| Job Level | {job_level} | {cluster_row['AvgJobLevel']:.1f} |
"""

    interventions = get_interventions(cluster_row["SegmentLabel"])
    interventions_md = "### 🩺 Recommended HR Retention Interventions\n" + "\n".join(
        f"- {item}" for item in interventions
    )

    compare_df = pd.DataFrame({
        "Metric": ["Age", "Monthly Income", "Job Satisfaction", "Work-Life Balance", "Years at Company"],
        "Simulated Employee": [age, monthly_income, job_satisfaction, work_life_balance, years_at_company],
        f"Cluster {cluster_id} Average": [
            round(cluster_row["AvgAge"], 1),
            round(cluster_row["AvgMonthlyIncome"], 0),
            round(cluster_row["AvgJobSatisfaction"], 2),
            round(cluster_row["AvgWorkLifeBalance"], 2),
            round(cluster_row["AvgYearsAtCompany"], 1),
        ],
    })
    compare_fig = px.bar(
        compare_df.melt(id_vars="Metric", var_name="Who", value_name="Value"),
        x="Metric", y="Value", color="Who", barmode="group",
        title="Simulated Employee vs. Cluster Persona",
    )
    compare_fig.update_layout(template="plotly_white")

    return result_md, interventions_md, compare_fig


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="IBM HR Analytics — Employee Segmentation") as demo:

    gr.Markdown(
        f"""
# 🧩 IBM HR Analytics — Employee Segmentation & Attrition Risk Explorer
### Strategic Initiative sponsored by Dr. Sarah Chen, Chief Human Resources Officer

This application applies **unsupervised K-Means clustering** to IBM's HR dataset
({TOTAL_EMPLOYEES:,} employees) to discover natural employee segments, surface attrition
risk patterns, and give HR business partners a tool to score hypothetical employee
profiles against real workforce personas — without using attrition as a training label.
"""
    )

    with gr.Tab("📊 Executive Overview"):
        gr.Markdown("## Key Performance Indicators")
        tab1_k_selector = gr.Dropdown(
            choices=K_RANGE, value=OPTIMAL_K, label="Cluster count (k) driving these insights",
        )
        kpi_output = gr.Markdown()
        gr.Markdown("## Dr. Chen's 5 Core Business Questions")
        questions_output = gr.Markdown()

        tab1_k_selector.change(
            fn=refresh_tab1, inputs=tab1_k_selector, outputs=[kpi_output, questions_output]
        )
        demo.load(fn=refresh_tab1, inputs=tab1_k_selector, outputs=[kpi_output, questions_output])

    with gr.Tab("📈 Model Evaluation & Validation"):
        gr.Markdown("## Clustering Quality Metrics (k = 2 to 10)")
        with gr.Row():
            elbow_plot = gr.Plot(value=build_elbow_chart(), label="Elbow Method")
            silhouette_plot = gr.Plot(value=build_silhouette_chart(), label="Silhouette Score")
        gr.Markdown(
            f"""
### How k was selected
- **Elbow Method:** Inertia (within-cluster sum of squares) decreases as k increases;
  the "elbow" marks the point of diminishing returns. Use the curve above to judge where
  additional clusters stop meaningfully reducing inertia.
- **Silhouette Score:** Measures how well-separated and internally cohesive clusters
  are (range -1 to 1; higher is better). This is computed for every k from 2–10 at
  startup, and the app's default k is simply whichever k scores highest — for the
  currently loaded dataset that is **k = {OPTIMAL_K}** (silhouette score =
  {ENGINE.silhouette_scores[OPTIMAL_K]:.3f}), marked with a star on both charts above.
- This default updates automatically if the underlying dataset changes — it is never
  fixed to a specific k. Analysts can still override it and explore any k from 2–10
  in the Cluster Profiles and Scenario Simulation tabs.
"""
        )

    with gr.Tab("🔍 Cluster Profiles & Segment Intelligence"):
        tab3_k_selector = gr.Dropdown(
            choices=K_RANGE, value=OPTIMAL_K, label="Select number of clusters (k)"
        )
        with gr.Row():
            attrition_plot = gr.Plot(label="Attrition Rate by Segment")
            income_plot = gr.Plot(label="Average Income by Segment")
        comparison_plot = gr.Plot(label="Satisfaction & Work-Life Balance by Segment")
        gr.Markdown("### Full Cluster Profile Table")
        profile_table = gr.Dataframe(interactive=False, wrap=True)

        tab3_k_selector.change(
            fn=refresh_tab3, inputs=tab3_k_selector,
            outputs=[profile_table, attrition_plot, comparison_plot, income_plot],
        )
        demo.load(
            fn=refresh_tab3, inputs=tab3_k_selector,
            outputs=[profile_table, attrition_plot, comparison_plot, income_plot],
        )

    with gr.Tab("🧪 Scenario Simulation & Risk Classifier"):
        gr.Markdown("## Simulate a Hypothetical Employee Profile")
        sim_k_selector = gr.Dropdown(
            choices=K_RANGE, value=OPTIMAL_K, label="Cluster model (k) to classify against"
        )
        with gr.Row():
            with gr.Column():
                age_in = gr.Slider(18, 60, value=28, step=1, label="Age")
                dept_in = gr.Dropdown(DEPARTMENTS, value="Research & Development", label="Department")
                gender_in = gr.Radio(GENDERS, value="Female", label="Gender")
                edu_in = gr.Dropdown([1, 2, 3, 4, 5], value=3, label="Education (1=Below College … 5=Doctor)")
                joblevel_in = gr.Dropdown([1, 2, 3, 4, 5], value=1, label="Job Level (1=Entry … 5=Executive)")
                income_in = gr.Slider(1000, 20000, value=3000, step=100, label="Monthly Income ($)")
            with gr.Column():
                satisfaction_in = gr.Radio([1, 2, 3, 4], value=1, label="Job Satisfaction (1=Low … 4=Very High)")
                wlb_in = gr.Radio([1, 2, 3, 4], value=1, label="Work-Life Balance (1=Bad … 4=Best)")
                totalyears_in = gr.Slider(0, 40, value=3, step=1, label="Total Working Years")
                yearsatco_in = gr.Slider(0, 40, value=2, step=1, label="Years at Company")
                yearsrole_in = gr.Slider(0, 18, value=1, step=1, label="Years in Current Role")
                yearsmgr_in = gr.Slider(0, 17, value=1, step=1, label="Years with Current Manager")

        simulate_btn = gr.Button("🚀 Simulate & Assign Segment", variant="primary")

        sim_result_md = gr.Markdown()
        sim_interventions_md = gr.Markdown()
        sim_compare_plot = gr.Plot()

        simulate_btn.click(
            fn=simulate_employee,
            inputs=[
                sim_k_selector, age_in, dept_in, gender_in, edu_in, joblevel_in,
                income_in, satisfaction_in, wlb_in, totalyears_in,
                yearsatco_in, yearsrole_in, yearsmgr_in,
            ],
            outputs=[sim_result_md, sim_interventions_md, sim_compare_plot],
        )

    gr.Markdown(
        "---\n*Built for IBM HR Analytics — unsupervised K-Means segmentation. "
        "Attrition is used only for post-hoc cluster profiling, never as a clustering input.*"
    )


if __name__ == "__main__":
    demo.launch()
