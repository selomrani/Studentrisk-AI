"""Streamlit UI for StudentRisk champion inference."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from studentrisk_ai.inference import (
    RISK_LABELS,
    StudentRiskPreprocessor,
    load_champion_model,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = ROOT / "data" / "raw" / "student_mental_health.csv"


def setting(name: str, default: str) -> str:
    """Read a setting from environment variables, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets[name])
    except (KeyError, FileNotFoundError):
        return default


@st.cache_resource
def load_champion():
    tracking_uri = setting("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = setting("MLFLOW_MODEL_NAME", "StudentRisk")
    model_alias = setting("MLFLOW_MODEL_ALIAS", "champion")
    return load_champion_model(tracking_uri, model_name, model_alias)


@st.cache_resource
def load_preprocessor() -> StudentRiskPreprocessor:
    if not REFERENCE_DATA.is_file():
        raise FileNotFoundError(f"Reference data was not found: {REFERENCE_DATA}")
    return StudentRiskPreprocessor(pd.read_csv(REFERENCE_DATA))


def build_input() -> pd.DataFrame:
    """Render widgets matching the raw training features."""
    with st.form("student-risk-form"):
        left, right = st.columns(2)
        with left:
            student_id = st.text_input("Student ID", value="NEW-STUDENT")
            age = st.number_input("Age", min_value=17, max_value=29, value=21)
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
            year = st.number_input("Year of Study", 1, 5, 3)
            sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, step=0.1)
            study = st.number_input("Study Hours per Week", 0.0, 100.0, 20.0)
            gpa = st.number_input("GPA", 0.0, 4.0, 3.0, step=0.01)
            social = st.number_input("Social Media Hours", 0.0, 24.0, 3.0, step=0.1)
        with right:
            screen = st.number_input("Screen Time Hours", 0.0, 24.0, 6.0, step=0.1)
            activity = st.selectbox("Physical Activity", ["High", "Low", "Moderate"])
            financial = st.number_input("Financial Stress", 0.0, 10.0, 5.0, step=0.1)
            family = st.number_input("Family Support", 0.0, 10.0, 5.0, step=0.1)
            attendance = st.number_input("Attendance Rate", 0.0, 100.0, 80.0)
            part_time = st.selectbox("Part-Time Job", ["No", "Yes"])
            diet = st.selectbox("Diet Quality", ["Average", "Good", "Poor"])
            history = st.selectbox("Previous Mental Health History", ["No", "Yes"])
        submitted = st.form_submit_button("Predict risk")

    return (
        pd.DataFrame(
            [{
                "Student_ID": student_id,
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year,
                "Sleep_Hours": sleep,
                "Study_Hours_Per_Week": study,
                "GPA": gpa,
                "Social_Media_Hours": social,
                "Screen_Time_Hours": screen,
                "Physical_Activity": activity,
                "Financial_Stress": financial,
                "Family_Support": family,
                "Attendance_Rate": attendance,
                "Part_Time_Job": part_time,
                "Diet_Quality": diet,
                "Previous_Mental_Health_History": history,
            }]
        ),
        submitted,
    )


st.set_page_config(page_title="StudentRisk", page_icon=":bar_chart:")
st.title("StudentRisk champion inference")
st.caption("Predictions are served by the champion model registered in MLflow.")

try:
    model = load_champion()
    preprocessor = load_preprocessor()
except Exception as error:
    st.error(f"Could not load the MLflow champion or preprocessing data: {error}")
    st.stop()

records, submitted = build_input()
if submitted:
    try:
        prediction = model.predict(preprocessor.transform(records))
        predicted_value = prediction[0]
        if hasattr(predicted_value, "item"):
            predicted_value = predicted_value.item()
        if isinstance(predicted_value, int) and 0 <= predicted_value < len(RISK_LABELS):
            predicted_value = RISK_LABELS[int(predicted_value)]
        st.success(f"Predicted mental-health risk: **{predicted_value}**")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(preprocessor.transform(records))[0]
            st.dataframe(
                pd.DataFrame(
                    {"Risk": RISK_LABELS[: len(probabilities)], "Probability": probabilities}
                ),
                hide_index=True,
            )
    except Exception as error:
        st.error(f"Prediction failed: {error}")
