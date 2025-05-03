import pandas as pd

# Load data
df = pd.read_csv("results/metrics.csv")

# 1. Overall averages per model
overall_model_avg = df.groupby("Model")[['MAE', 'RMSE', 'MAPE', 'Train R2', 'Test R2']].mean().round(2)
print("\nOverall averages per model:\n")
print(overall_model_avg.to_string())

# 2. Per-station, per-model summary
grouped_stats = df.groupby(['Station', 'Model'])[['MAE', 'RMSE', 'MAPE']].agg(['mean', 'std']).round(2)
print("\nPer-station, per-model performance (mean ± std):\n")
print(grouped_stats.to_string())

# 3. Best model per station based on lowest mean MAE
best_models = df.groupby(['Station', 'Model'])['MAE'].mean().reset_index()
best_per_station = best_models.loc[best_models.groupby("Station")["MAE"].idxmin()]
print("\nBest model per station (based on lowest MAE):\n")
print(best_per_station.to_string(index=False))

# Step-wise performance per model with compact column names
model_abbr = {
    "Decision Tree": "DT",
    "Gradient Boosting": "GB",
    "KNN": "KNN",
    "Linear Regression": "LR",
    "Random Forest": "RF",
    "SVM": "SVM"
}

# Compute the pivoted stats
stepwise_raw = df.groupby(["Step", "Model"])[['MAE', 'RMSE', 'R2 Step']].mean().unstack()

# Rename columns like ('MAE', 'Random Forest') → 'MAE_RF'
compact_columns = []
for metric, model in stepwise_raw.columns:
    short_model = model_abbr.get(model, model.replace(" ", ""))
    short_metric = "R2" if metric == "R2 Step" else metric
    compact_columns.append(f"{short_metric}_{short_model}")

stepwise_clean = stepwise_raw.copy()
stepwise_clean.columns = compact_columns
stepwise_clean = stepwise_clean.round(2)

print("\nStep-wise performance per model:\n")
print(stepwise_clean.to_string(index=True))


# 5. Model ranking based on mean metrics
ranking = df.groupby("Model")[['MAE', 'RMSE', 'MAPE']].mean().rank().sort_values("MAE")
print("\nModel ranking (1 = best, based on average MAE / RMSE / MAPE):\n")
print(ranking.to_string())

# 6. Correlation matrix
correlation = df[['MAE', 'RMSE', 'MAPE', 'EVS', 'Test R2', 'Train R2']].corr().round(2)
print("\nCorrelation matrix between metrics:\n")
print(correlation.to_string())