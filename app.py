from __future__ import annotations

from html import escape
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

N2O_HTML = "N<sub>2</sub>O"
O2_HTML = "O<sub>2</sub>"
NH4_HTML = "NH<sub>4</sub><sup>+</sup>"
NO3_HTML = "NO<sub>3</sub><sup>-</sup>"
PO4_HTML = "PO<sub>4</sub><sup>3-</sup>"
R2_HTML = "R<sup>2</sup>"


st.set_page_config(page_title=PAGE_TITLE, page_icon=":chart_with_upwards_trend:", layout="wide")


def apply_style() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            color: {INK};
        }}
        sub, sup {{
            font-size: 0.68em;
            line-height: 0;
            position: relative;
            vertical-align: baseline;
        }}
        sub {{ bottom: -0.28em; }}
        sup {{ top: -0.46em; }}
        .stApp {{ background: {BG}; }}
        .block-container {{ padding-top: 1.05rem; padding-bottom: 2.4rem; max-width: 1320px; }}
        div[data-testid="stVerticalBlock"] {{ gap: 0.58rem; }}
        .hero {{
            background: linear-gradient(135deg, #173F45 0%, #2F6F73 52%, #5EA7A3 100%);
            color: white;
            padding: 1.28rem 1.72rem;
            border-radius: 10px;
            margin-bottom: 0.62rem;
            box-shadow: 0 12px 30px rgba(24, 66, 72, 0.15);
        }}
        .hero h1 {{
            color: #FFFFFF !important;
            font-size: 1.92rem;
            line-height: 1.1;
            margin: 0 0 0.32rem 0;
            font-weight: 800;
            letter-spacing: 0;
        }}
        .hero h1 sub, .hero h1 sup {{ color: #FFFFFF !important; }}
        .hero p {{
            margin: 0;
            color: rgba(255,255,255,0.92) !important;
            font-size: 0.94rem;
            line-height: 1.42;
            max-width: 920px;
        }}
        .hero p sub, .hero p sup {{ color: rgba(255,255,255,0.92) !important; }}
        .card {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 8px 26px rgba(36,50,63,0.06);
        }}
        .metric-card {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-left: 4px solid {PRIMARY};
            border-radius: 8px;
            padding: 0.66rem 0.88rem;
            min-height: 78px;
            box-shadow: 0 6px 18px rgba(36,50,63,0.045);
        }}
        .metric-label {{ color: {MUTED}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.035em; line-height: 1.22; }}
        .metric-value {{ color: {INK}; font-size: 1.34rem; font-weight: 800; margin-top: 0.16rem; line-height: 1.12; }}
        .small-note {{ color: {MUTED}; font-size: 0.8rem; line-height: 1.38; margin: 0.04rem 0 0.1rem 0; }}
        .result-box {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 10px 30px rgba(36,50,63,0.07);
            display: grid;
            grid-template-columns: minmax(210px, 0.85fr) minmax(300px, 1.55fr);
            gap: 1.35rem;
            align-items: start;
        }}
        .result-text {{ color: {INK}; font-size: 0.94rem; line-height: 1.45; margin: 0 0 0.72rem 0; }}
        .threshold-value {{ color: {INK}; font-size: 2.05rem; font-weight: 800; line-height: 1.05; margin: 0.2rem 0 0.95rem 0; }}
        .result-message {{
            border-radius: 8px;
            padding: 0.78rem 0.92rem;
            font-size: 0.9rem;
            line-height: 1.42;
            font-weight: 500;
        }}
        .result-message-ok {{ background: #E7F5EF; color: #23734E; }}
        .result-message-warn {{ background: #FFF4D7; color: #735817; }}
        .result-value {{ font-size: 2.55rem; color: {PRIMARY}; font-weight: 800; line-height: 1; margin-top: 0.22rem; }}
        .result-unit {{ color: {MUTED}; font-size: 0.96rem; font-weight: 700; line-height: 1.25; margin: 0.22rem 0 0.55rem 0; }}
        .input-label {{
            color: {INK};
            font-size: 0.9rem;
            font-weight: 700;
            line-height: 1.16;
            min-height: 1.05rem;
            display: flex;
            align-items: flex-end;
            margin: 0.02rem 0 0.08rem 0;
        }}
        .input-label sub, .input-label sup {{
            font-size: 0.72em;
        }}
        .status-pill {{
            display: inline-block;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.82rem;
        }}
        .pill-normal {{ background: #E7F5EF; color: #23734E; }}
        .pill-high {{ background: #FBECEA; color: #A63C32; }}
        .pill-caution {{ background: #FFF4D7; color: #8B6B1D; }}
        div[data-testid="stMetricValue"] {{ font-weight: 800; color: {INK}; }}
        h2 {{ color: {INK}; font-size: 1.2rem !important; line-height: 1.22 !important; padding-bottom: 0 !important; margin: 0.34rem 0 0.08rem 0 !important; }}
        h3 {{ color: {INK}; font-size: 1rem !important; line-height: 1.25 !important; margin-top: 0.48rem !important; }}
        h4 {{ color: {INK}; font-size: 0.95rem !important; line-height: 1.2 !important; margin: 0.46rem 0 0.16rem 0 !important; }}
        label, .stNumberInput label, .stFileUploader label {{
            color: {INK} !important;
            font-size: 0.82rem !important;
            font-weight: 650 !important;
            line-height: 1.18 !important;
            min-height: 1.95rem;
            display: flex !important;
            align-items: flex-end !important;
        }}
        .stNumberInput input {{
            font-size: 0.92rem !important;
            min-height: 2.18rem;
        }}
        div[data-testid="stNumberInput"] label {{
            display: none !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stNumberInput"] {{
            margin-top: -0.45rem;
        }}
        div[data-testid="stCaptionContainer"] {{
            color: {MUTED};
            font-size: 0.84rem;
            line-height: 1.42;
        }}
        div[data-testid="stTabs"] button p {{
            font-size: 0.92rem;
            font-weight: 700;
        }}
        div[data-testid="stForm"] {{
            background: white;
            border: 1px solid rgba(36,50,63,0.08);
            border-radius: 8px;
            padding: 0.72rem 0.92rem 0.9rem 0.92rem;
            box-shadow: 0 6px 18px rgba(36,50,63,0.045);
        }}
        div[data-testid="stHorizontalBlock"] {{
            gap: 0.78rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: 8px;
            font-weight: 700;
            border: 1px solid rgba(36,50,63,0.14);
            min-height: 2.35rem;
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
            .hero {{ padding: 1.12rem 1.1rem; margin-bottom: 0.55rem; }}
            .hero h1 {{ font-size: 1.52rem; }}
            .metric-value {{ font-size: 1.28rem; }}
            .result-value {{ font-size: 2.15rem; }}
            .result-box {{ grid-template-columns: 1fr; gap: 1rem; }}
            .input-label {{ min-height: auto; }}
            label, .stNumberInput label, .stFileUploader label {{ min-height: auto; }}
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


def labeled_number_input(label_html: str, key: str, *args, **kwargs):
    st.markdown(f"<div class='input-label'>{label_html}</div>", unsafe_allow_html=True)
    return st.number_input(" ", *args, key=key, label_visibility="collapsed", **kwargs)


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
    st.markdown(
        f"<div class='small-note'>Current-hour {N2O_HTML} soft sensing using current process values and recent {N2O_HTML} history.</div>",
        unsafe_allow_html=True,
    )

    with st.form("single_prediction_form"):
        st.markdown(f"#### {N2O_HTML} history", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        target = bundle["target"]
        med_n2o = default_raw(bundle, target)
        with c1:
            n2o_lag1 = labeled_number_input("N₂O, t-1 h", "n2o_lag1h", 0.0, 10.0, med_n2o, 0.01)
        with c2:
            n2o_lag2 = labeled_number_input("N₂O, t-2 h", "n2o_lag2h", 0.0, 10.0, med_n2o, 0.01)
        with c3:
            n2o_lag3 = labeled_number_input("N₂O, t-3 h", "n2o_lag3h", 0.0, 10.0, med_n2o, 0.01)
        with c4:
            n2o_roll3 = labeled_number_input("N₂O rolling 3 h", "n2o_roll3h", 0.0, 10.0, med_n2o, 0.01)
        with c5:
            n2o_roll6 = labeled_number_input("N₂O rolling 6 h", "n2o_roll6h", 0.0, 10.0, med_n2o, 0.01)

        st.markdown("#### Nitrogen and aeration")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nh4 = labeled_number_input("NH₄⁺ (mg L⁻¹)", "nh4_mg_l", 0.0, 30.0, default_raw(bundle, "NH4_T1"), 0.1)
            no3 = labeled_number_input("NO₃⁻ (mg L⁻¹)", "no3_mg_l", 0.0, 60.0, default_raw(bundle, "NO3_T1"), 0.1)
        with c2:
            po4 = labeled_number_input("PO₄³⁻ (mg L⁻¹)", "po4_mg_l", 0.0, 20.0, default_raw(bundle, "PO4_T1"), 0.1)
            o2 = labeled_number_input("Dissolved O₂ (mg L⁻¹)", "o2_mg_l", 0.0, 10.0, default_raw(bundle, "O2_T1"), 0.05)
        with c3:
            o2_setpoint = labeled_number_input("O₂ setpoint", "o2_setpoint", 0.0, 5.0, default_raw(bundle, "O2_SP_T1"), 0.05)
            airflow = labeled_number_input("Airflow", "airflow", 0.0, 8000.0, default_raw(bundle, "AIRFLOW_T1"), 50.0)
        with c4:
            valve = labeled_number_input("Valve position (%)", "valve_position", 0.0, 100.0, default_raw(bundle, "VALVE_T1"), 1.0)
            ss = labeled_number_input("Suspended solids", "suspended_solids", 0.0, 10.0, default_raw(bundle, "SS_T1"), 0.05)

        st.markdown("#### Load, temperature and time")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temperature = labeled_number_input("Temperature (°C)", "temperature_c", 0.0, 40.0, default_raw(bundle, "TEMP_T1"), 0.1)
        with c2:
            influent_q = labeled_number_input("Influent Q", "influent_q", 0.0, 15000.0, default_raw(bundle, "INLET_Q"), 50.0)
        with c3:
            stormwater_flow = labeled_number_input("Stormwater flow", "stormwater_flow", 0.0, 2.0, default_raw(bundle, "SWM_INLET_FLOW"), 0.05)
        with c4:
            hour = labeled_number_input("Hour of day", "hour_of_day", 0, 23, 12, 1)
            day_of_year = labeled_number_input("Day of year", "day_of_year", 1, 366, 180, 1)

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

    if domain_warnings:
        message_class = "result-message-warn"
        message = "Applicability-domain warnings:<br>" + "<br>".join(f"- {escape(w)}" for w in domain_warnings)
    else:
        message_class = "result-message-ok"
        message = "Inputs are within the training 1-99% range for the displayed process variables."

    st.markdown(
        f"""
        <div class="result-box">
            <div>
                <div class="metric-label">Predicted dissolved {N2O_HTML}</div>
                <div class="result-value">{pred:.3f}</div>
                <div class="result-unit">mg L<sup>-1</sup></div>
                <span class="status-pill {pill_class}">{label}</span>
            </div>
            <div>
                <p class="result-text">The threshold is the 95th percentile of the training-period {N2O_HTML} distribution.</p>
                <div class="metric-label">High-emission threshold</div>
                <div class="threshold-value">{threshold:.3f} mg L<sup>-1</sup></div>
                <div class="result-message {message_class}">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        metric_card(f"Test {R2_HTML}", fmt(test["r2"], 3), BLUE)
    with c2:
        metric_card("Test RMSE", fmt(test["rmse"], 3), TEAL, f"log1p({N2O_HTML})")
    with c3:
        metric_card("Test MAE", fmt(test["mae"], 3), GOLD, f"log1p({N2O_HTML})")
    with c4:
        metric_card("Train rows", f"{bundle['training_rows']:,}", CORAL)

    st.markdown("#### Scientific scope")
    st.markdown(
        f"This app is a current-hour soft sensor for dissolved {N2O_HTML} at Avedore WWTP. "
        "It follows the manuscript feature engineering and uses the regularized HGB model. "
        f"Predictions rely strongly on recent {N2O_HTML} history, so missing or unrealistic history inputs reduce reliability.",
        unsafe_allow_html=True,
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
            <h1>N<sub>2</sub>O Soft Sensor for Wastewater Treatment</h1>
            <p>Current-hour dissolved N<sub>2</sub>O prediction using a leakage-audited machine-learning model trained on full-scale hourly data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Model", "Regularized HGB", PRIMARY)
    with c2:
        metric_card(f"Hold-out {R2_HTML}", fmt(bundle["test_metrics"]["r2"], 3), BLUE)
    with c3:
        metric_card("High-emission threshold", f"{bundle['high_emission_threshold_mg_l']:.3f} mg L<sup>-1</sup>", CORAL)

    tab1, tab2, tab3 = st.tabs(["Prediction", "Batch upload", "Model information"])
    with tab1:
        render_single_prediction(bundle)
    with tab2:
        render_batch_prediction(bundle)
    with tab3:
        render_model_info(bundle)


if __name__ == "__main__":
    main()
