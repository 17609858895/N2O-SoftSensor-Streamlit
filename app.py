from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "model" / "model_bundle.joblib"
EXAMPLE_PATH = APP_DIR / "data" / "example_input.csv"
TEMPLATE_PATH = APP_DIR / "data" / "single_prediction_template.csv"

PAGE_TITLE = "N₂O Soft Sensor"
PRIMARY = "#2F6F73"
TEAL = "#43A7A4"
BLUE = "#4D7EA8"
CORAL = "#D96C5F"
GOLD = "#C9A227"
INK = "#24323F"
MUTED = "#657487"
BG = "#F6F8F7"


st.set_page_config(page_title=PAGE_TITLE, page_icon=":chart_with_upwards_trend:", layout="wide")


def apply_style() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
        .stApp {{ background: {BG}; }}
        .block-container {{ padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1240px; }}
        .hero {{
            background: linear-gradient(135deg, #173F45 0%, #2F6F73 52%, #5EA7A3 100%);
            color: white;
            padding: 2rem 2.3rem;
            border-radius: 10px;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(24, 66, 72, 0.18);
        }}
        .hero h1 {{ font-size: 2.25rem; line-height: 1.12; margin: 0 0 0.45rem 0; font-weight: 800; letter-spacing: 0; }}
        .hero p {{ margin: 0; color: rgba(255,255,255,0.88); font-size: 1rem; line-height: 1.55; max-width: 880px; }}
        .card {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 26px rgba(36,50,63,0.06);
        }}
        .metric-card {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-left: 4px solid {PRIMARY};
            border-radius: 8px;
            padding: 0.85rem 1rem;
            min-height: 98px;
            box-shadow: 0 8px 24px rgba(36,50,63,0.05);
        }}
        .metric-label {{ color: {MUTED}; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; line-height: 1.25; }}
        .metric-value {{ color: {INK}; font-size: 1.55rem; font-weight: 800; margin-top: 0.24rem; line-height: 1.15; }}
        .small-note {{ color: {MUTED}; font-size: 0.84rem; line-height: 1.45; margin-top: 0.12rem; }}
        .result-box {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 10px 30px rgba(36,50,63,0.07);
        }}
        .result-value {{ font-size: 2.7rem; color: {PRIMARY}; font-weight: 800; line-height: 1; margin-top: 0.25rem; }}
        .status-pill {{
            display: inline-block;
            padding: 0.35rem 0.68rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.84rem;
        }}
        .pill-normal {{ background: #E7F5EF; color: #23734E; }}
        .pill-high {{ background: #FBECEA; color: #A63C32; }}
        .pill-caution {{ background: #FFF4D7; color: #8B6B1D; }}
        div[data-testid="stMetricValue"] {{ font-weight: 800; color: {INK}; }}
        h2 {{ color: {INK}; font-size: 1.35rem !important; line-height: 1.25 !important; padding-bottom: 0.15rem !important; }}
        h3 {{ color: {INK}; font-size: 1.05rem !important; line-height: 1.3 !important; margin-top: 0.8rem !important; }}
        h4 {{ color: {INK}; font-size: 0.96rem !important; line-height: 1.25 !important; margin: 0.75rem 0 0.35rem 0 !important; }}
        label, .stNumberInput label, .stFileUploader label {{
            color: {INK} !important;
            font-size: 0.86rem !important;
            font-weight: 650 !important;
            line-height: 1.25 !important;
        }}
        .stNumberInput input {{
            font-size: 0.94rem !important;
            min-height: 2.4rem;
        }}
        div[data-testid="stCaptionContainer"] {{
            color: {MUTED};
            font-size: 0.88rem;
            line-height: 1.45;
        }}
        div[data-testid="stTabs"] button p {{
            font-size: 0.95rem;
            font-weight: 700;
        }}
        div[data-testid="stForm"] {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 0.95rem 1.05rem 1.1rem 1.05rem;
            box-shadow: 0 8px 24px rgba(36,50,63,0.05);
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            font-weight: 700;
            border: 1px solid rgba(36,50,63,0.14);
        }}
        .stButton > button[kind="primary"] {{
            background: {PRIMARY};
            border-color: {PRIMARY};
        }}
        .stAlert {{
            border-radius: 8px;
        }}
        @media (max-width: 760px) {{
            .block-container {{ padding-top: 1rem; }}
            .hero {{ padding: 1.35rem 1.25rem; }}
            .hero h1 {{ font-size: 1.65rem; }}
            .metric-value {{ font-size: 1.35rem; }}
            .result-value {{ font-size: 2.25rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        st.error("Model bundle is missing. Run scripts/train_export_model.py before deployment.")
        st.stop()
    return joblib.load(BUNDLE_PATH)


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def metric_card(label: str, value: str, accent: str = PRIMARY, note: str | None = None) -> None:
    note_html = f"<div class='small-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{accent}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(value_mg_l: float, threshold: float) -> tuple[str, str]:
    if value_mg_l >= threshold:
        return "High-emission threshold exceeded", "pill-high"
    if value_mg_l >= threshold * 0.65:
        return "Near high-emission threshold", "pill-caution"
    return "Below high-emission threshold", "pill-normal"


def default_raw(bundle: dict, raw_col: str, stat: str = "median") -> float:
    return float(bundle["raw_stats"].get(raw_col, {}).get(stat, 0.0))


def check_domain(bundle: dict, guided: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    raw_map = bundle["raw_input_map"]
    for guided_col, raw_col in raw_map.items():
        if guided_col not in guided:
            continue
        stats = bundle["raw_stats"].get(raw_col)
        if not stats:
            continue
        value = float(guided[guided_col])
        if value < stats["p01"] or value > stats["p99"]:
            warnings.append(
                f"{guided_col}={value:.3g} is outside the training 1-99% range "
                f"({stats['p01']:.3g}-{stats['p99']:.3g})."
            )
    target_stats = bundle["raw_stats"].get(bundle["target"], {})
    for col in ("n2o_lag1h", "n2o_lag2h", "n2o_lag3h", "n2o_roll3h", "n2o_roll6h"):
        value = float(guided.get(col, 0.0))
        if target_stats and (value < target_stats["p01"] or value > target_stats["p99"]):
            warnings.append(
                f"{col}={value:.3g} is outside the N₂O training 1-99% range "
                f"({target_stats['p01']:.3g}-{target_stats['p99']:.3g})."
            )
    return warnings


def build_features_from_guided(bundle: dict, guided: dict[str, float]) -> pd.DataFrame:
    row = dict(bundle["feature_medians"])
    raw_map = bundle["raw_input_map"]

    for guided_col, raw_col in raw_map.items():
        if guided_col not in guided:
            continue
        value = float(guided[guided_col])
        row[f"{raw_col}_current"] = value
        for lag in (1, 2, 3):
            row[f"{raw_col}_lag{lag}h"] = value
        row[f"{raw_col}_roll3h"] = value
        row[f"{raw_col}_roll6h"] = value
        row[f"{raw_col}_missing_lag1h"] = 0.0

    n2o_history = {
        "N2O_T1_lag1h": guided.get("n2o_lag1h"),
        "N2O_T1_lag2h": guided.get("n2o_lag2h"),
        "N2O_T1_lag3h": guided.get("n2o_lag3h"),
        "N2O_T1_roll3h": guided.get("n2o_roll3h"),
        "N2O_T1_roll6h": guided.get("n2o_roll6h"),
    }
    fallback_n2o = default_raw(bundle, bundle["target"])
    for feature, value in n2o_history.items():
        row[feature] = float(fallback_n2o if value is None else value)

    hour = int(guided.get("hour", 12))
    doy = int(guided.get("day_of_year", 180))
    row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    row["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    row["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))

    return pd.DataFrame([row], columns=bundle["feature_columns"])


def predict_features(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    pred_log = bundle["model"].predict(features[bundle["feature_columns"]])
    return np.clip(np.expm1(pred_log), a_min=0, a_max=None)


def read_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return output.getvalue()


def batch_predict(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = bundle["feature_columns"]
    guided_cols = bundle["guided_input_columns"]
    messages: list[str] = []

    if set(feature_cols).issubset(df.columns):
        features = df[feature_cols].copy()
        output = df.copy()
        messages.append("Detected engineered feature columns; predictions used the exact model feature matrix.")
    elif set(guided_cols).issubset(df.columns):
        feature_rows = []
        for _, row in df.iterrows():
            guided = {col: float(row[col]) for col in guided_cols}
            feature_rows.append(build_features_from_guided(bundle, guided).iloc[0])
        features = pd.DataFrame(feature_rows, columns=feature_cols)
        output = df.copy()
        messages.append("Detected guided input columns; missing engineered features were filled from training medians.")
    else:
        missing_guided = [c for c in guided_cols if c not in df.columns]
        raise ValueError(
            "Uploaded file must contain either all engineered feature columns or the guided input template columns. "
            f"Missing guided columns: {', '.join(missing_guided[:8])}"
        )

    preds = predict_features(bundle, features)
    threshold = float(bundle["high_emission_threshold_mg_l"])
    output["predicted_n2o_mg_l"] = preds
    output["high_emission_flag"] = np.where(preds >= threshold, "yes", "no")
    output["threshold_mg_l"] = threshold
    return output, messages


def render_single_prediction(bundle: dict) -> None:
    st.subheader("Single prediction")
    st.caption("Current-hour N₂O soft sensing using current process values and recent N₂O history.")

    with st.form("single_prediction_form"):
        st.markdown("#### N₂O history")
        c1, c2, c3, c4, c5 = st.columns(5)
        target = bundle["target"]
        med_n2o = default_raw(bundle, target)
        with c1:
            n2o_lag1 = st.number_input("N₂O, t−1 h", 0.0, 10.0, med_n2o, 0.01)
        with c2:
            n2o_lag2 = st.number_input("N₂O, t−2 h", 0.0, 10.0, med_n2o, 0.01)
        with c3:
            n2o_lag3 = st.number_input("N₂O, t−3 h", 0.0, 10.0, med_n2o, 0.01)
        with c4:
            n2o_roll3 = st.number_input("N₂O rolling 3 h", 0.0, 10.0, med_n2o, 0.01)
        with c5:
            n2o_roll6 = st.number_input("N₂O rolling 6 h", 0.0, 10.0, med_n2o, 0.01)

        st.markdown("#### Nitrogen and aeration")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nh4 = st.number_input("NH₄⁺ (mg/L)", 0.0, 30.0, default_raw(bundle, "NH4_T1"), 0.1)
            no3 = st.number_input("NO₃⁻ (mg/L)", 0.0, 60.0, default_raw(bundle, "NO3_T1"), 0.1)
        with c2:
            po4 = st.number_input("PO₄³⁻ (mg/L)", 0.0, 20.0, default_raw(bundle, "PO4_T1"), 0.1)
            o2 = st.number_input("Dissolved O₂ (mg/L)", 0.0, 10.0, default_raw(bundle, "O2_T1"), 0.05)
        with c3:
            o2_setpoint = st.number_input("O₂ setpoint", 0.0, 5.0, default_raw(bundle, "O2_SP_T1"), 0.05)
            airflow = st.number_input("Airflow", 0.0, 8000.0, default_raw(bundle, "AIRFLOW_T1"), 50.0)
        with c4:
            valve = st.number_input("Valve position (%)", 0.0, 100.0, default_raw(bundle, "VALVE_T1"), 1.0)
            ss = st.number_input("Suspended solids", 0.0, 10.0, default_raw(bundle, "SS_T1"), 0.05)

        st.markdown("#### Load, temperature and time")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temperature = st.number_input("Temperature (°C)", 0.0, 40.0, default_raw(bundle, "TEMP_T1"), 0.1)
        with c2:
            influent_q = st.number_input("Influent Q", 0.0, 15000.0, default_raw(bundle, "INLET_Q"), 50.0)
        with c3:
            stormwater_flow = st.number_input("Stormwater flow", 0.0, 2.0, default_raw(bundle, "SWM_INLET_FLOW"), 0.05)
        with c4:
            hour = st.number_input("Hour of day", 0, 23, 12, 1)
            day_of_year = st.number_input("Day of year", 1, 366, 180, 1)

        submitted = st.form_submit_button("Predict N₂O", type="primary", use_container_width=True)

    if not submitted:
        st.info("Enter process values and click Predict N₂O.")
        return

    guided = {
        "n2o_lag1h": n2o_lag1,
        "n2o_lag2h": n2o_lag2,
        "n2o_lag3h": n2o_lag3,
        "n2o_roll3h": n2o_roll3,
        "n2o_roll6h": n2o_roll6,
        "nh4": nh4,
        "no3": no3,
        "po4": po4,
        "o2": o2,
        "o2_setpoint": o2_setpoint,
        "airflow": airflow,
        "valve": valve,
        "ss": ss,
        "temperature": temperature,
        "influent_q": influent_q,
        "stormwater_flow": stormwater_flow,
        "hour": hour,
        "day_of_year": day_of_year,
    }
    features = build_features_from_guided(bundle, guided)
    pred = float(predict_features(bundle, features)[0])
    threshold = float(bundle["high_emission_threshold_mg_l"])
    label, pill_class = risk_label(pred, threshold)
    domain_warnings = check_domain(bundle, guided)

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 2])
    with left:
        st.markdown("<div class='metric-label'>Predicted dissolved N₂O</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-value'>{pred:.3f}</div>", unsafe_allow_html=True)
        st.caption("mg/L")
        st.markdown(f"<span class='status-pill {pill_class}'>{label}</span>", unsafe_allow_html=True)
    with right:
        st.write("The threshold is the 95th percentile of the training-period N₂O distribution.")
        st.metric("High-emission threshold", f"{threshold:.3f} mg/L")
        if domain_warnings:
            st.warning("Applicability-domain warnings:\n\n" + "\n".join(f"- {w}" for w in domain_warnings))
        else:
            st.success("Inputs are within the training 1-99% range for the displayed process variables.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_batch_prediction(bundle: dict) -> None:
    st.subheader("Batch prediction")
    st.caption("Upload CSV/XLSX using the guided template, or upload a file containing all engineered model features.")

    c1, c2 = st.columns(2)
    if TEMPLATE_PATH.exists():
        c1.download_button(
            "Download single-row template",
            TEMPLATE_PATH.read_bytes(),
            file_name="single_prediction_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if EXAMPLE_PATH.exists():
        c2.download_button(
            "Download example input",
            EXAMPLE_PATH.read_bytes(),
            file_name="example_input.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded = st.file_uploader("Upload input file", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        return

    try:
        df = read_upload(uploaded)
        st.write("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        result, messages = batch_predict(bundle, df)
    except Exception as exc:
        st.error(str(exc))
        return

    for message in messages:
        st.info(message)
    st.write("Predictions")
    st.dataframe(result.head(50), use_container_width=True)
    st.download_button(
        "Download N₂O predictions as CSV",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="n2o_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download N₂O predictions as Excel",
        dataframe_to_excel_bytes(result),
        file_name="n2o_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def render_model_info(bundle: dict) -> None:
    st.subheader("Model and scope")
    train = bundle["train_metrics"]
    test = bundle["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Test R²", fmt(test["r2"], 3), BLUE)
    with c2:
        metric_card("Test RMSE", fmt(test["rmse"], 3), TEAL, "log1p(N₂O)")
    with c3:
        metric_card("Test MAE", fmt(test["mae"], 3), GOLD, "log1p(N₂O)")
    with c4:
        metric_card("Train rows", f"{bundle['training_rows']:,}", CORAL)

    st.markdown("#### Scientific scope")
    st.write(
        "This app is a current-hour soft sensor for dissolved N₂O at Avedore WWTP. "
        "It follows the manuscript feature engineering and uses the regularized HGB model. "
        "Predictions rely strongly on recent N₂O history, so missing or unrealistic history inputs reduce reliability."
    )
    st.markdown("#### Caveats")
    for caveat in bundle["caveats"]:
        st.write(f"- {caveat}")

    st.markdown("#### Training period")
    st.write(f"{bundle['time_min_utc']} to {bundle['time_max_utc']}; split: {bundle['split']}.")


def main() -> None:
    apply_style()
    bundle = load_bundle()

    st.markdown(
        """
        <div class="hero">
            <h1>N₂O Soft Sensor for Wastewater Treatment</h1>
            <p>Current-hour dissolved N₂O prediction using a leakage-audited machine-learning model trained on full-scale hourly data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model", "Regularized HGB", PRIMARY)
    with c2:
        metric_card("Hold-out R²", fmt(bundle["test_metrics"]["r2"], 3), BLUE)
    with c3:
        metric_card("High-emission threshold", f"{bundle['high_emission_threshold_mg_l']:.3f} mg/L", CORAL)

    tab1, tab2, tab3 = st.tabs(["Prediction", "Batch upload", "Model information"])
    with tab1:
        render_single_prediction(bundle)
    with tab2:
        render_batch_prediction(bundle)
    with tab3:
        render_model_info(bundle)


if __name__ == "__main__":
    main()
