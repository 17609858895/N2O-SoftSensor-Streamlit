from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_PATH = PROJECT_ROOT / "aved_raw.csv"
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"

TARGET = "N2O_T1"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8
HIGH_EMISSION_THRESHOLD_MG_L = 0.5872536966881963

RENAME = {
    "BIOLOGY.BLOWERSTATION 1.Q.AIRFLOW value": "AIRFLOW_BLOWER",
    "BIOLOGY.LINE 3 TANK 1 VALVE 1.PCT value": "VALVE_T1",
    "BIOLOGY.LINE 3 TANK 1.N2O value": "N2O_T1",
    "BIOLOGY.LINE 3 TANK 1.NH4 value": "NH4_T1",
    "BIOLOGY.LINE 3 TANK 1.NO3 value": "NO3_T1",
    "BIOLOGY.LINE 3 TANK 1.O2 value": "O2_T1",
    "BIOLOGY.LINE 3 TANK 1.O2.SETPOINT value": "O2_SP_T1",
    "BIOLOGY.LINE 3 TANK 1.PROCESSPHASE value": "PHASE_T1",
    "BIOLOGY.LINE 3 TANK 1.Q.AIRFLOW value": "AIRFLOW_T1",
    "BIOLOGY.LINE 3 TANK 1.SS value": "SS_T1",
    "BIOLOGY.LINE 3 TANK 1.TEMPERATURE value": "TEMP_T1",
    "BIOLOGY.LINE 3 TANK 2 VALVE 1.PCT value": "VALVE_T2",
    "BIOLOGY.LINE 3 TANK 2.O2 value": "O2_T2",
    "BIOLOGY.LINE 3 TANK 2.O2.SETPOINT value": "O2_SP_T2",
    "BIOLOGY.LINE 3 TANK 2.PROCESSPHASE value": "PHASE_T2",
    "BIOLOGY.LINE 3 TANK 2.Q.AIRFLOW value": "AIRFLOW_T2",
    "BIOLOGY.LINE 3 TANK 2.SS value": "SS_T2",
    "BIOLOGY.LINE 3 TANK 2.TEMPERATURE value": "TEMP_T2",
    "BIOLOGY.LINE 3.PHASECODE.SETPOINT value": "PHASECODE_SP",
    "BIOLOGY.LINE 3.PROCESSPHASE.INLET TANK value": "INLET_TANK_PHASE",
    "BIOLOGY.LINE 3.PROCESSPHASE.OUTLET TANK value": "OUTLET_TANK_PHASE",
    "BIOLOGY.LINE 3 TANK 1.PO4 value": "PO4_T1",
    "INLET.Q value": "INLET_Q",
    "INLET.STATE.SWM INLET FLOW value": "SWM_INLET_FLOW",
}

GUIDED_INPUT_COLUMNS = [
    "n2o_lag1h",
    "n2o_lag2h",
    "n2o_lag3h",
    "n2o_roll3h",
    "n2o_roll6h",
    "nh4",
    "no3",
    "po4",
    "o2",
    "o2_setpoint",
    "airflow",
    "valve",
    "ss",
    "temperature",
    "influent_q",
    "stormwater_flow",
    "hour",
    "day_of_year",
]

RAW_INPUT_MAP = {
    "nh4": "NH4_T1",
    "no3": "NO3_T1",
    "po4": "PO4_T1",
    "o2": "O2_T1",
    "o2_setpoint": "O2_SP_T1",
    "airflow": "AIRFLOW_T1",
    "valve": "VALVE_T1",
    "ss": "SS_T1",
    "temperature": "TEMP_T1",
    "influent_q": "INLET_Q",
    "stormwater_flow": "SWM_INLET_FLOW",
}


def load_hourly_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RENAME)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    df[TARGET] = df[TARGET].clip(lower=0)
    hourly = df.resample("1h").mean()
    hourly = hourly.dropna(subset=[TARGET])
    return hourly


def make_features(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = hourly.copy()
    missing_flags = df.isna().astype(float)
    interpolated = df.interpolate(method="time", limit=12).ffill().bfill()

    feature_data: dict[str, pd.Series | np.ndarray] = {}
    op_cols = [c for c in interpolated.columns if c != TARGET]
    for col in op_cols:
        feature_data[f"{col}_current"] = interpolated[col]
        for lag in (1, 2, 3):
            feature_data[f"{col}_lag{lag}h"] = interpolated[col].shift(lag)
        feature_data[f"{col}_roll3h"] = interpolated[col].shift(1).rolling(3, min_periods=2).mean()
        feature_data[f"{col}_roll6h"] = interpolated[col].shift(1).rolling(6, min_periods=3).mean()

    for lag in (1, 2, 3):
        feature_data[f"{TARGET}_lag{lag}h"] = interpolated[TARGET].shift(lag)
    feature_data[f"{TARGET}_roll3h"] = interpolated[TARGET].shift(1).rolling(3, min_periods=2).mean()
    feature_data[f"{TARGET}_roll6h"] = interpolated[TARGET].shift(1).rolling(6, min_periods=3).mean()

    for col in op_cols:
        feature_data[f"{col}_missing_lag1h"] = missing_flags[col].shift(1)

    idx = interpolated.index
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    feature_data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feature_data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feature_data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feature_data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    features = pd.DataFrame(feature_data, index=interpolated.index)
    y = np.log1p(interpolated[TARGET])
    valid = features.notna().all(axis=1) & y.notna()
    return features.loc[valid], y.loc[valid]


def split_time(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cut = int(len(X) * TRAIN_FRACTION)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def make_regularized_hgb() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.035,
                    max_iter=260,
                    max_leaf_nodes=12,
                    max_depth=3,
                    min_samples_leaf=120,
                    l2_regularization=10.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metrics_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def raw_stats(hourly: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in sorted(set(RAW_INPUT_MAP.values()) | {TARGET}):
        if col not in hourly.columns:
            continue
        s = hourly[col].dropna()
        stats[col] = {
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return stats


def guided_from_feature_row(row: pd.Series, timestamp: pd.Timestamp) -> dict[str, float]:
    values = {
        "n2o_lag1h": row.get("N2O_T1_lag1h", np.nan),
        "n2o_lag2h": row.get("N2O_T1_lag2h", np.nan),
        "n2o_lag3h": row.get("N2O_T1_lag3h", np.nan),
        "n2o_roll3h": row.get("N2O_T1_roll3h", np.nan),
        "n2o_roll6h": row.get("N2O_T1_roll6h", np.nan),
        "hour": int(timestamp.hour),
        "day_of_year": int(timestamp.dayofyear),
    }
    for guided_col, raw_col in RAW_INPUT_MAP.items():
        values[guided_col] = row.get(f"{raw_col}_current", np.nan)
    return {k: float(v) for k, v in values.items()}


def write_templates(bundle: dict, X_test: pd.DataFrame, y_test: pd.Series, test_pred: np.ndarray) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stats = bundle["raw_stats"]
    single = {
        "n2o_lag1h": stats[TARGET]["median"],
        "n2o_lag2h": stats[TARGET]["median"],
        "n2o_lag3h": stats[TARGET]["median"],
        "n2o_roll3h": stats[TARGET]["median"],
        "n2o_roll6h": stats[TARGET]["median"],
        "hour": 12,
        "day_of_year": 180,
    }
    for guided_col, raw_col in RAW_INPUT_MAP.items():
        single[guided_col] = stats[raw_col]["median"]
    pd.DataFrame([single], columns=GUIDED_INPUT_COLUMNS).to_csv(DATA_DIR / "single_prediction_template.csv", index=False)

    rows = []
    for i, (idx, row) in enumerate(X_test.head(20).iterrows()):
        guided = guided_from_feature_row(row, idx)
        guided["observed_log1p_n2o"] = float(y_test.loc[idx])
        guided["observed_n2o_mg_l"] = float(np.expm1(y_test.loc[idx]))
        guided["model_prediction_n2o_mg_l"] = float(np.expm1(test_pred[i]))
        rows.append(guided)
    pd.DataFrame(rows).to_csv(DATA_DIR / "example_input.csv", index=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_data(DATA_PATH)
    X, y = make_features(hourly)
    X_train, X_test, y_train, y_test = split_time(X, y)

    model = make_regularized_hgb()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = metrics_dict(y_train, train_pred)
    test_metrics = metrics_dict(y_test, test_pred)
    train_metrics["gap"] = float(train_metrics["r2"] - test_metrics["r2"])

    feature_medians = X_train.median(numeric_only=True).to_dict()
    stats = raw_stats(hourly)
    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "feature_medians": {k: float(v) for k, v in feature_medians.items()},
        "raw_stats": stats,
        "target": TARGET,
        "target_transform": "log1p(N2O_T1)",
        "prediction_unit": "mg/L dissolved N2O",
        "guided_input_columns": GUIDED_INPUT_COLUMNS,
        "raw_input_map": RAW_INPUT_MAP,
        "high_emission_threshold_mg_l": HIGH_EMISSION_THRESHOLD_MG_L,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_rows": int(len(X)),
        "hourly_rows": int(len(hourly)),
        "time_min_utc": str(hourly.index.min()),
        "time_max_utc": str(hourly.index.max()),
        "split": "first 80% train, last 20% test",
        "caveats": [
            "This is a current-hour soft sensor, not a next-hour forecasting model.",
            "Recent N2O history is a dominant information source; predictions are less reliable without valid history inputs.",
            "Operating-window outputs are associative screening hypotheses and should not be used as causal control set-points.",
            "Online nitrite was not available in the source data.",
        ],
    }

    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib", compress=3)
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_templates(bundle, X_test, y_test, test_pred)

    print(json.dumps({"test_metrics": test_metrics, "model_rows": len(X)}, indent=2))


if __name__ == "__main__":
    main()
