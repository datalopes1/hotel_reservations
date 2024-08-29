import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_and_model(data_path, model_path):
    try:
        df = pd.read_csv(data_path)
        model_series = pd.read_pickle(model_path)
        logging.info("Data and model loaded.")
        return df, model_series
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

def make_predictions(df, model_series):
    try:
        X = df[model_series['features']]
        y_proba = model_series['model'].predict_proba(X)[:,1]
        y_pred = (y_proba > 0.60).astype(int)
        df['pred_status'] = y_pred
        df['pred_status'] = df['pred_status'].map({1: 'Canceled', 0: 'Not_Canceled'})
        df['cancel_prob'] = y_proba
        logging.info("Predictions done.")
        return df
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

def save_results(df, output_path):
    try:
        results = df[['Booking_ID', 'booking_status', 'pred_status', 'cancel_prob']].copy()
        results.to_excel(output_path, index=False)
        logging.info(f"Results saved in {output_path}.")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data/raw/Hotel Reservations.csv"
    model_path = base_dir / "models/classifier.pkl"
    output_path = base_dir / "data/processed/model_predictions.xlsx"

    df, model_series = load_data_and_model(data_path, model_path)
    df = make_predictions(df, model_series)
    save_results(df, output_path)

if __name__ == "__main__":
    main()