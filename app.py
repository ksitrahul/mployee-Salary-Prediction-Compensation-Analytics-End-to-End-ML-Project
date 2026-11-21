import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

st.set_page_config(
    page_title="Compensation Intelligence & Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom subtle CSS for KPI cards
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.18);
    padding: 14px 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "salary_model.joblib"
DATA_PATH = "employee_salary_dataset.csv"

@st.cache_resource
def load_or_train_bundle():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH).drop(columns=["Employee_ID"])
    df.loc[(df["Age"] < 18) | (df["Age"] > 65), "Age"] = np.nan
    X = df.drop(columns=["Annual_Salary_LPA"])
    y = df["Annual_Salary_LPA"]

    num_cols = ["Age", "Experience_Years", "Performance_Rating", "Certifications", "Overtime_Hours", "Company_Tenure", "Projects_Completed", "Skill_Score"]
    cat_cols = ["Gender", "Education", "Department", "Job_Level", "Remote_Work", "City"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    pipe = Pipeline([("preprocessor", preprocessor), ("regressor", HuberRegressor(max_iter=1500, epsilon=1.35))]).fit(X_train, y_train)
    low_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=200, max_depth=4, random_state=42))]).fit(X_train, y_train)
    up_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=200, max_depth=4, random_state=42))]).fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mask = y_test <= 100
    benchmarks = df.groupby(["Department", "Job_Level"])["Annual_Salary_LPA"].agg(
        count="count",
        p25=lambda s: round(float(np.percentile(s, 25)), 2),
        median=lambda s: round(float(np.median(s)), 2),
        p75=lambda s: round(float(np.percentile(s, 75)), 2),
        mean=lambda s: round(float(np.mean(s)), 2)
    ).reset_index()

    bundle = {
        "pipeline": pipe,
        "lower_pipeline": low_pipe,
        "upper_pipeline": up_pipe,
        "model_name": "Huber Regressor (Outlier-Robust)",
        "features": num_cols + cat_cols,
        "metrics": {
            "Standard Band R2": round(float(r2_score(y_test[mask], preds[mask])), 4),
            "Standard Band MAE": round(float(mean_absolute_error(y_test[mask], preds[mask])), 3),
            "Overall Median AE": round(float(median_absolute_error(y_test, preds)), 3)
        },
        "benchmarks": benchmarks.to_dict(orient="records")
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.loc[(df["Age"] < 18) | (df["Age"] > 65), "Age"] = np.nan
        return df
    return None

bundle = load_or_train_bundle()
pipeline = bundle["pipeline"]
lower_pipe = bundle["lower_pipeline"]
upper_pipe = bundle["upper_pipeline"]
metrics = bundle.get("metrics", {})
benchmarks_df = pd.DataFrame(bundle.get("benchmarks", []))
df_data = load_dataset()

# Sidebar
st.sidebar.title("💼 Compensation Studio")
view = st.sidebar.radio(
    "Navigation",
    [
        "Salary Estimator & What-If Simulator",
        "HR Batch Payroll & Equity Audit",
        "Workforce Compensation Explorer",
        "Model Benchmarks & Drivers"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Production Model Stats")
st.sidebar.write(f"**Model:** {bundle.get('model_name', 'Huber Regressor')}")
if "Standard Band R2" in metrics:
    st.sidebar.metric("R² (Standard Salary Band)", f"{metrics['Standard Band R2']:.4f}")
if "Standard Band MAE" in metrics:
    st.sidebar.metric("Mean Abs. Error (MAE)", f"₹{metrics['Standard Band MAE']:.2f} LPA")
if "Overall Median AE" in metrics:
    st.sidebar.metric("Median Abs. Error", f"₹{metrics['Overall Median AE']:.2f} LPA")

st.title("Enterprise Salary Prediction & Compensation Analytics")
st.caption("ML-powered compensation benchmarking, 80% confidence interval estimation, career growth simulation, and HR pay-parity auditing.")

# ---------------------------------------------------------
# VIEW 1: INDIVIDUAL SALARY ESTIMATOR & WHAT-IF SIMULATOR
# ---------------------------------------------------------
if view == "Salary Estimator & What-If Simulator":
    with st.form("salary_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Role & Organization")
            department = st.selectbox("Department", ["IT", "Operations", "Marketing", "Sales", "HR"])
            job_level = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead", "Manager"], index=1)
            city = st.selectbox("Work Location (City)", ["Mumbai", "Hyderabad", "Delhi", "Chennai"])
            remote_work = st.selectbox("Remote Work", ["Yes", "No"])

        with c2:
            st.subheader("Experience & Education")
            age = st.number_input("Age (Years)", min_value=18, max_value=65, value=32, step=1)
            experience = st.number_input("Total Experience (Years)", min_value=0.0, max_value=40.0, value=7.0, step=0.5)
            tenure = st.number_input("Company Tenure (Years)", min_value=0, max_value=30, value=3, step=1)
            education = st.selectbox("Highest Education", ["Bachelor", "Master", "PhD", "Diploma"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        with c3:
            st.subheader("Performance & Skills")
            skill_score = st.slider("Technical / Domain Skill Score", min_value=50, max_value=100, value=80)
            perf_rating = st.slider("Performance Rating (1-5)", min_value=1, max_value=5, value=4)
            certifications = st.number_input("Certifications Earned", min_value=0, max_value=15, value=3, step=1)
            projects = st.number_input("Projects Completed", min_value=1, max_value=40, value=14, step=1)
            overtime = st.number_input("Monthly Overtime Hours", min_value=0, max_value=80, value=20, step=1)

        submitted = st.form_submit_button("Estimate Compensation & Run What-If Scenarios", type="primary", use_container_width=True)

    # Compute on initial load and on submit so the page is immediately interactive
    profile = {
        "Age": float(age),
        "Gender": gender,
        "Education": education,
        "Experience_Years": float(experience),
        "Department": department,
        "Job_Level": job_level,
        "Performance_Rating": int(perf_rating),
        "Certifications": int(certifications),
        "Overtime_Hours": int(overtime),
        "Remote_Work": remote_work,
        "City": city,
        "Company_Tenure": int(tenure),
        "Projects_Completed": int(projects),
        "Skill_Score": float(skill_score)
    }
    input_df = pd.DataFrame([profile])

    pred_lpa = float(pipeline.predict(input_df)[0])
    low_lpa = float(min(lower_pipe.predict(input_df)[0], pred_lpa - 1.5))
    high_lpa = float(max(upper_pipe.predict(input_df)[0], pred_lpa + 1.5))
    monthly_gross_inr = (pred_lpa * 100000) / 12.0

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Annual CTC", f"₹{pred_lpa:.2f} LPA")
    m2.metric("80% Confidence Band", f"₹{low_lpa:.1f} – ₹{high_lpa:.1f} LPA")
    m3.metric("Estimated Monthly Gross", f"₹{monthly_gross_inr:,.0f}")

    peer_row = benchmarks_df[(benchmarks_df["Department"] == department) & (benchmarks_df["Job_Level"] == job_level)]
    if not peer_row.empty:
        peer_med = float(peer_row.iloc[0]["median"])
        delta_pct = ((pred_lpa - peer_med) / peer_med) * 100
        m4.metric(f"vs. {department} {job_level} Median", f"₹{peer_med:.2f} LPA", f"{delta_pct:+.1f}%")

    # Career Growth / What-If Simulator
    st.subheader("Career Growth & Compensation Uplift Simulator")
    levels = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    curr_idx = levels.index(job_level)
    next_level = levels[min(curr_idx + 1, len(levels) - 1)]

    scenarios = []
    # Scenario A: +2 Years Experience & +10 Skill Score
    s_upskill = profile.copy()
    s_upskill["Experience_Years"] = min(40.0, experience + 2.0)
    s_upskill["Skill_Score"] = min(100.0, float(skill_score) + 10.0)
    s_upskill["Certifications"] = certifications + 2
    val_upskill = float(pipeline.predict(pd.DataFrame([s_upskill]))[0])
    scenarios.append({"Scenario": "Upskill (+2 Yrs Exp, +10 Skill, +2 Certs)", "Projected LPA": round(val_upskill, 2), "Uplift (LPA)": round(val_upskill - pred_lpa, 2)})

    # Scenario B: Promotion to Next Job Level
    if next_level != job_level:
        s_promo = profile.copy()
        s_promo["Job_Level"] = next_level
        s_promo["Experience_Years"] = min(40.0, experience + 2.0)
        val_promo = float(pipeline.predict(pd.DataFrame([s_promo]))[0])
        scenarios.append({"Scenario": f"Promotion to {next_level} (+2 Yrs Exp)", "Projected LPA": round(val_promo, 2), "Uplift (LPA)": round(val_promo - pred_lpa, 2)})

    # Scenario C: Higher Degree (Master / PhD)
    if education in ["Diploma", "Bachelor", "Master"]:
        next_edu = {"Diploma": "Bachelor", "Bachelor": "Master", "Master": "PhD"}[education]
        s_edu = profile.copy()
        s_edu["Education"] = next_edu
        val_edu = float(pipeline.predict(pd.DataFrame([s_edu]))[0])
        scenarios.append({"Scenario": f"Complete {next_edu} Degree", "Projected LPA": round(val_edu, 2), "Uplift (LPA)": round(val_edu - pred_lpa, 2)})

    # Scenario D: Switch to IT in Mumbai
    if department != "IT" or city != "Mumbai":
        s_switch = profile.copy()
        s_switch["Department"] = "IT"
        s_switch["City"] = "Mumbai"
        val_switch = float(pipeline.predict(pd.DataFrame([s_switch]))[0])
        scenarios.append({"Scenario": "Transition to IT Role in Mumbai", "Projected LPA": round(val_switch, 2), "Uplift (LPA)": round(val_switch - pred_lpa, 2)})

    scen_df = pd.DataFrame(scenarios)
    sc_col1, sc_col2 = st.columns([1.2, 1])
    with sc_col1:
        st.dataframe(scen_df, use_container_width=True, hide_index=True)
    with sc_col2:
        st.bar_chart(scen_df.set_index("Scenario")["Projected LPA"])

# ---------------------------------------------------------
# VIEW 2: HR BATCH PAYROLL & EQUITY AUDIT
# ---------------------------------------------------------
elif view == "HR Batch Payroll & Equity Audit":
    st.subheader("Batch Compensation Prediction & Pay-Parity Audit")
    st.markdown(
        "Upload an employee CSV file. If `Annual_Salary_LPA` is included in the CSV, the system automatically flags **Underpaid (Retention Risk)** and **Above-Band Outlier** records."
    )

    uploaded = st.file_uploader("Upload Employee CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "Age" in batch_df.columns:
            batch_df.loc[(batch_df["Age"] < 18) | (batch_df["Age"] > 65), "Age"] = np.nan

        req_cols = bundle["features"]
        missing = [c for c in req_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
        else:
            preds = pipeline.predict(batch_df[req_cols])
            lows = lower_pipe.predict(batch_df[req_cols])
            highs = upper_pipe.predict(batch_df[req_cols])

            out = batch_df.copy()
            out["Predicted_Salary_LPA"] = np.round(preds, 2)
            out["Band_Low_LPA"] = np.round(np.minimum(lows, preds - 1.5), 2)
            out["Band_High_LPA"] = np.round(np.maximum(highs, preds + 1.5), 2)

            if "Annual_Salary_LPA" in out.columns:
                out["Variance_LPA"] = np.round(out["Annual_Salary_LPA"] - out["Predicted_Salary_LPA"], 2)
                out["Audit_Status"] = np.where(
                    out["Annual_Salary_LPA"] < out["Band_Low_LPA"],
                    "Underpaid (Retention Risk)",
                    np.where(out["Annual_Salary_LPA"] > out["Band_High_LPA"], "Above Market Band", "Aligned with Market")
                )
                k1, k2, k3 = st.columns(3)
                k1.metric("Total Audited", len(out))
                k2.metric("Retention Risk Count", int((out["Audit_Status"] == "Underpaid (Retention Risk)").sum()))
                k3.metric("Above Market Count", int((out["Audit_Status"] == "Above Market Band").sum()))

            st.dataframe(out.head(100), use_container_width=True)
            st.download_button(
                "Download Full Payroll Audit CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="salary_compensation_audit.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------
# VIEW 3: WORKFORCE COMPENSATION EXPLORER
# ---------------------------------------------------------
elif view == "Workforce Compensation Explorer":
    st.subheader("Interactive Workforce Compensation Analytics")
    if df_data is not None:
        f1, f2, f3 = st.columns(3)
        sel_depts = f1.multiselect("Filter Department", sorted(df_data["Department"].unique()), default=sorted(df_data["Department"].unique()))
        sel_levels = f2.multiselect("Filter Job Level", ["Junior", "Mid", "Senior", "Lead", "Manager"], default=["Junior", "Mid", "Senior", "Lead", "Manager"])
        sel_cities = f3.multiselect("Filter City", sorted(df_data["City"].unique()), default=sorted(df_data["City"].unique()))

        filtered = df_data[
            df_data["Department"].isin(sel_depts) &
            df_data["Job_Level"].isin(sel_levels) &
            df_data["City"].isin(sel_cities)
        ]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Filtered Headcount", f"{len(filtered):,}")
        s2.metric("Median Salary", f"₹{filtered['Annual_Salary_LPA'].median():.2f} LPA")
        s3.metric("Mean Salary", f"₹{filtered['Annual_Salary_LPA'].mean():.2f} LPA")
        s4.metric("Avg Experience", f"{filtered['Experience_Years'].mean():.1f} Yrs")

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Median Salary by Department (LPA)")
            dept_med = filtered.groupby("Department")["Annual_Salary_LPA"].median().sort_values(ascending=False)
            st.bar_chart(dept_med)
        with c_right:
            st.markdown("#### Median Salary by Job Level (LPA)")
            lvl_med = filtered.groupby("Job_Level")["Annual_Salary_LPA"].median().reindex(["Junior", "Mid", "Senior", "Lead", "Manager"]).dropna()
            st.bar_chart(lvl_med)

        st.markdown("#### Department × Job Level Compensation Benchmark Table (LPA)")
        st.dataframe(benchmarks_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# VIEW 4: MODEL BENCHMARKS & DRIVERS
# ---------------------------------------------------------
elif view == "Model Benchmarks & Drivers":
    st.subheader("Model Evaluation & Feature Impact Weights")
    all_res = bundle.get("all_results", [])
    if all_res:
        st.markdown("### Comparative Model Benchmark")
        st.dataframe(pd.DataFrame(all_res), use_container_width=True, hide_index=True)

    st.markdown("### Top Positive & Negative Salary Drivers (Huber Coefficients)")
    num_cols = ["Age", "Experience_Years", "Performance_Rating", "Certifications", "Overtime_Hours", "Company_Tenure", "Projects_Completed", "Skill_Score"]
    cat_cols = ["Gender", "Education", "Department", "Job_Level", "Remote_Work", "City"]
    ohe_names = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    feat_names = num_cols + list(ohe_names)
    coefs = pipeline.named_steps["regressor"].coef_

    coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient_LPA": coefs}).sort_values("Coefficient_LPA", ascending=False)
    st.bar_chart(coef_df.set_index("Feature")["Coefficient_LPA"])
