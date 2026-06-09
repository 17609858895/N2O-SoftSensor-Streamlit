# N2O Soft Sensor Streamlit App

This repository contains a deployable Streamlit interface for the dissolved N2O soft-sensing model developed from full-scale wastewater treatment data.

## What the app does

- Predicts current-hour dissolved N2O concentration from process inputs and recent N2O history.
- Supports single-sample prediction through an interactive form.
- Supports batch prediction from CSV or Excel files.
- Flags high-emission conditions using the 95th percentile of the training-period N2O distribution.
- Reports applicability-domain warnings when user inputs fall outside the training 1-99% range.

## Model

The deployed model is the regularized HistGradientBoostingRegressor pipeline used in the manuscript workflow. It is trained with chronological validation: the first 80% of hourly records are used for training and the last 20% for testing. The target is log1p-transformed dissolved N2O and predictions are back-transformed to mg/L.

The raw dataset is not included in this repository. The app loads the exported model bundle in `model/model_bundle.joblib`, together with metadata and input templates.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use these settings:

- Repository: this GitHub repository
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

No API key or Streamlit secret is required.

## Files

- `app.py`: Streamlit web interface.
- `model/model_bundle.joblib`: trained model and feature metadata.
- `model/model_metadata.json`: readable model metadata.
- `data/single_prediction_template.csv`: single-row upload template.
- `data/example_input.csv`: example batch input.
- `scripts/train_export_model.py`: reproducible model export script. It expects `aved_raw.csv` in the parent research folder and is not required for the online app.
