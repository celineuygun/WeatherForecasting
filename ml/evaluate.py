import os
import pandas as pd

variables = ["temperature_c", "humidity", "wind_speed"]
base_path = "results"

model_abbr = {
    "Decision Tree": "DT",
    "Gradient Boosting": "GB",
    "KNN": "KNN",
    "Linear Regression": "LR",
    "Random Forest": "RF",
    "SVM": "SVM",
    "LightGBM": "LGBM",
    "XGBoost": "XGB"
}

for var in variables:
    print(f"\n{'='*25} {var.upper()} {'='*25}")

    filepath = os.path.join(base_path, f"metrics_{var}.csv")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        continue

    df = pd.read_csv(filepath)

    # 1. Overall averages per model
    overall_model_avg = df.groupby("Model")[['MAE', 'RMSE', 'MAPE', 'Train R2', 'Test R2']].mean().round(2)
    print("\n1. Overall averages per model:\n")
    print(overall_model_avg.to_string())

    # 2. Per-station, per-model summary
    grouped_stats = df.groupby(['Station', 'Model'])[['MAE', 'RMSE', 'MAPE']].agg(['mean', 'std']).round(2)
    print("\n2. Per-station, per-model performance (mean ± std):\n")
    print(grouped_stats.to_string())

    # 3. Best model per station based on lowest mean MAE
    best_models = df.groupby(['Station', 'Model'])['MAE'].mean().reset_index()
    best_per_station = best_models.loc[best_models.groupby("Station")["MAE"].idxmin()]
    print("\n3. Best model per station (based on lowest MAE):\n")
    print(best_per_station.to_string(index=False))

    # 4. Step-wise performance per model
    stepwise_raw = df.groupby(["Step", "Model"])[['MAE', 'RMSE', 'R2 Step']].mean().unstack()
    compact_columns = []
    for metric, model in stepwise_raw.columns:
        short_model = model_abbr.get(model, model.replace(" ", ""))
        short_metric = "R2" if metric == "R2 Step" else metric
        compact_columns.append(f"{short_metric}_{short_model}")

    stepwise_clean = stepwise_raw.copy()
    stepwise_clean.columns = compact_columns
    stepwise_clean = stepwise_clean.round(2)

    print("\n4. Step-wise performance per model:\n")
    print(stepwise_clean.to_string(index=True))

    # 5. Model ranking
    ranking = df.groupby("Model")[['MAE', 'RMSE', 'MAPE']].mean().rank().sort_values("MAE")
    print("\n5. Model ranking (lower = better):\n")
    print(ranking.to_string())

    # 6. Correlation between metrics
    correlation = df[['MAE', 'RMSE', 'MAPE', 'EVS', 'Test R2', 'Train R2']].corr().round(2)
    print("\n6. Correlation matrix between metrics:\n")
    print(correlation.to_string())
