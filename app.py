# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Branding / Page ---
st.set_page_config(
    page_title="SurviveAI — Titanic Survival Prediction",
    page_icon="🛟",
    layout="centered"
)

# --- Model Loader (prefiere SurviveAI, cae a Titanic si existe) ---
@st.cache_resource(show_spinner=False)
def load_model():
    paths = [
        "models/surviveai_model.joblib",
        "./models/surviveai_model.joblib",
        "../models/surviveai_model.joblib",
        "../../models/surviveai_model.joblib",
        # fallback (compatibilidad)
        "models/titanic_model.joblib",
        "./models/titanic_model.joblib",
        "../models/titanic_model.joblib",
        "../../models/titanic_model.joblib",
    ]
    for p in paths:
        if Path(p).exists():
            return joblib.load(p)
    raise FileNotFoundError("Model not found. Expected models/surviveai_model.joblib")

model = load_model()

# --- Header ---
st.title("SurviveAI — Titanic Survival Prediction")
st.caption("Explainable ML to estimate survival probability given passenger context.")

# --- Feature Engineering (replica del entrenamiento) ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["family_size"] = out["sibsp"] + out["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)
    out["fare_per_person"] = out["fare"] / out["family_size"].replace(0, 1)
    # extrae título estilo 'Last, Title. First'
    out["title"] = out["name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    out["title"] = (
        out["title"]
        .replace(["Mlle", "Ms"], "Miss")
        .replace("Mme", "Mrs")
        .fillna("Mr")
    )
    return out

# --- UI Form ---
with st.form("form"):
    st.subheader("Passenger Profile")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Class (pclass)", [1, 2, 3], index=2)
        sex = st.selectbox("Sex (sex)", ["male", "female"], index=0)
        age = st.slider("Age (age)", 0, 80, 29)
        sibsp = st.number_input("SibSp (siblings/spouses)", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Parch (parents/children)", min_value=0, max_value=10, value=0, step=1)
    with col2:
        fare = st.number_input("Fare (fare)", min_value=0.0, max_value=600.0, value=32.20, step=0.1)
        embarked = st.selectbox("Embarked (embarked)", ["S", "C", "Q"], index=0)
        has_cabin_ui = st.checkbox("Has cabin?", value=False)
        name = st.text_input("Name (Last, Title. First)", "Doe, Mr. John")
        ticket = st.text_input("Ticket", "A/5 21171")

    submitted = st.form_submit_button("Predict")

# 1) En build_input_df() -> quita passengerid del dict inicial
def build_input_df():
    base = pd.DataFrame([{
        # "passengerid": np.nan,   # <-- elimina esta línea
        "pclass": int(pclass),
        "sex": str(sex),
        "age": float(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": float(fare),
        "embarked": str(embarked),
        "name": str(name),
        "ticket": str(ticket),
        "has_cabin": int(has_cabin_ui),
    }])

    base = add_features(base)

    # Optional flags si existían en tu entrenamiento
    for col_flag in ["is_out_age", "is_out_fare"]:
        if col_flag not in base.columns:
            base[col_flag] = 0

    # Descubre columnas esperadas por el preprocesador del modelo
    try:
        expected = list(model.named_steps["prep"].feature_names_in_)
    except Exception:
        expected = None

    # Compatibilidad: algunos pipelines usaban 'Has_Cabin'
    if expected and "Has_Cabin" in expected and "has_cabin" in base.columns:
        base["Has_Cabin"] = base["has_cabin"]

    # Si el pipeline esperara 'passengerid'/'PassengerId', añade un placeholder interno
    if expected:
        if "passengerid" in expected and "passengerid" not in base.columns:
            base["passengerid"] = 0
        if "PassengerId" in expected and "PassengerId" not in base.columns:
            base["PassengerId"] = 0

    # Coherencia simple título/sexo
    if base.loc[0, "sex"] == "male" and base.loc[0, "title"] in ["Miss", "Mrs"]:
        base.loc[0, "title"] = "Mr"
    if base.loc[0, "sex"] == "female" and base.loc[0, "title"] == "Mr":
        base.loc[0, "title"] = "Mrs"

    return base

# --- Inference ---
if submitted:
    df_input = build_input_df()
    try:
        pred = int(model.predict(df_input)[0])
        proba_fn = getattr(model, "predict_proba", None)
        prob_survive = float(proba_fn(df_input)[0][1]) if proba_fn else None

        st.subheader("Result")
        if pred == 1:
            st.success("🟩 Survives")
        else:
            st.error("🟥 Does not survive")

        if prob_survive is not None:
            st.write(f"Survival probability: {prob_survive*100:.2f}%")

        with st.expander("Model input (engineered features)"):
            st.dataframe(df_input)

    except Exception as e:
        st.exception(e)
