import os
import pandas as pd
import argparse
from utils import MODEL_ABBR

# Get per_station from command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--per_station", action="store_true", help="Enable per-station mode")
args = parser.parse_args()
per_station = args.per_station

mode = "per_station" if per_station else "merged"
base_path = os.path.join("results", mode)
metrics_path = os.path.join(base_path, "metrics.csv")

if not os.path.exists(metrics_path):
    print(f"ERROR: File not found: {metrics_path}")
    exit(1)

df = pd.read_csv(metrics_path)
variables = df['Variable'].unique()

for var in variables:
    print(f"\n{'='*25} {var.upper()} {'='*25}")
    df_var = df[df["Variable"] == var].copy()

    # 1. Overall averages per model
    overall_model_avg = df_var.groupby("Model")[['MAE', 'RMSE', 'MAPE', 'Train R2', 'Test R2']].mean().round(2)
    print("\nOverall averages per model:\n")
    print(overall_model_avg.to_string())

    # 2. Per-station, per-model summary (only if multiple stations exist)
    if per_station:
        grouped_stats = df_var.groupby(['Station', 'Model'])[['MAE', 'RMSE', 'MAPE']].agg(['mean', 'std']).round(2)
        print("\nPer-station, per-model performance (mean ± std):\n")
        print(grouped_stats.to_string())

        # 3. Best model per station
        best_models = df_var.groupby(['Station', 'Model'])['MAE'].mean().reset_index()
        best_per_station = best_models.loc[best_models.groupby("Station")["MAE"].idxmin()]
        print("\nBest model per station (based on lowest MAE):\n")
        print(best_per_station.to_string(index=False))

    # 4. Step-wise performance per model
    stepwise_raw = df_var.groupby(["Step", "Model"])[['MAE', 'RMSE', 'R2 Step']].mean().unstack()
    compact_columns = []
    for metric, model in stepwise_raw.columns:
        short_model = MODEL_ABBR.get(model, model.replace(" ", ""))
        short_metric = "R2" if metric == "R2 Step" else metric
        compact_columns.append(f"{short_metric}_{short_model}")

    stepwise_clean = stepwise_raw.copy()
    stepwise_clean.columns = compact_columns
    stepwise_clean = stepwise_clean.round(2)

    print("\nStep-wise performance per model:\n")
    print(stepwise_clean.to_string(index=True))

    # 5. Model ranking
    ranking = df_var.groupby("Model")[['MAE', 'RMSE', 'MAPE']].mean().rank().sort_values("MAE")
    print("\nModel ranking (lower = better):\n")
    print(ranking.to_string())

    # 6. Metric correlations
    correlation = df_var[['MAE', 'RMSE', 'MAPE', 'EVS', 'Test R2', 'Train R2']].corr().round(2)
    print("\nCorrelation matrix between metrics:\n")
    print(correlation.to_string())
