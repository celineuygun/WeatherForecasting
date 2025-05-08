import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from pathlib import Path
import numpy as np
import warnings

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)

def visualize_predictions(variables=["temperature_c", "humidity", "wind_speed"], per_station=True):
    base_dir = "results/per_station" if per_station else "results/merged"

    for variable in variables:
        predictions_file = f"{base_dir}/predictions_{variable}.csv"
        if not os.path.exists(predictions_file):
            print(f"ERROR: File not found: {predictions_file}")
            continue

        print(f"\nINFO: Visualizing: {predictions_file}")
        preds_df = pd.read_csv(predictions_file)

        if "Station" not in preds_df.columns:
            preds_df["Station"] = "merged"

        plot_base_dir = Path(base_dir) / "plots" / variable
        plot_base_dir.mkdir(parents=True, exist_ok=True)

        stations = preds_df['Station'].unique()
        models = preds_df['Model'].unique()
        steps = sorted(preds_df['Step'].unique())

        global_density_max = 0
        for station in stations:
            df_station = preds_df[preds_df['Station'] == station]
            for model in models:
                for step in steps:
                    subset = df_station[
                        (df_station['Model'] == model) &
                        (df_station['Step'] == step)
                    ]
                    if subset.empty:
                        continue
                    hist = sns.histplot(subset["Predicted"] - subset["True"], bins=30, stat="density", kde=True)
                    lines = hist.get_lines()
                    if lines:
                        y_data = lines[0].get_data()[1]
                        max_density = np.nanmax(y_data)
                        global_density_max = max(global_density_max, max_density)
                    plt.clf()

        for station in stations:
            station_dir = plot_base_dir / f"station_{station}"
            station_dir.mkdir(exist_ok=True)

            df_station = preds_df[preds_df['Station'] == station].copy()
            df_station["residual"] = df_station["Predicted"] - df_station["True"]

            val_min = df_station[["True", "Predicted"]].min().min()
            val_max = df_station[["True", "Predicted"]].max().max()
            residual_min = df_station["residual"].min()
            residual_max = df_station["residual"].max()

            for model in models:
                model_dir = station_dir / model.replace(" ", "_")
                model_dir.mkdir(exist_ok=True)

                for step in steps:
                    subset = df_station[
                        (df_station['Model'] == model) &
                        (df_station['Step'] == step)
                    ].reset_index(drop=True)

                    if subset.empty:
                        continue

                    rolling_mae = np.abs(subset["residual"]).rolling(window=100, min_periods=1).mean()
                    mae_max = rolling_mae.max()

                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    fig.suptitle(f"{variable} | Station {station} - {model} (t+{step})", fontsize=14)

                    # 1. Prediction vs True
                    axs[0, 0].plot(subset['True'].values, label='True', alpha=0.7)
                    axs[0, 0].plot(subset['Predicted'].values, label='Predicted', alpha=0.7)
                    axs[0, 0].set_ylabel(variable)
                    axs[0, 0].legend()
                    axs[0, 0].set_title("Prediction vs. True")

                    # 2. Rolling MAE
                    axs[0, 1].plot(rolling_mae)
                    axs[0, 1].set_ylim(0, mae_max * 1.1)
                    axs[0, 1].set_title("Rolling MAE (window=100)")
                    axs[0, 1].set_ylabel("MAE")

                    # 3. Residual Histogram (fixed y-axis)
                    sns.histplot(subset["residual"], bins=30, kde=True, ax=axs[1, 0], color="salmon", stat="density")
                    axs[1, 0].set_xlim(residual_min, residual_max)
                    axs[1, 0].set_ylim(0, global_density_max * 1.1)
                    axs[1, 0].set_xlabel("Predicted - True")
                    axs[1, 0].set_title("Residuals Histogram")

                    # 4. True vs Predicted Scatter
                    r_val = r2_score(subset['True'], subset['Predicted']) if len(subset) > 1 else 0
                    sns.scatterplot(x="True", y="Predicted", data=subset, alpha=0.3, ax=axs[1, 1], color="crimson")
                    axs[1, 1].plot([val_min, val_max], [val_min, val_max], linestyle="--", color="black")
                    axs[1, 1].set_xlim(val_min, val_max)
                    axs[1, 1].set_ylim(val_min, val_max)
                    axs[1, 1].text(val_min + 1, val_max - 5, f"R² = {r_val:.2f}", fontsize=10)
                    axs[1, 1].set_title("True vs Predicted Scatter")
                    axs[1, 1].set_xlabel("True")
                    axs[1, 1].set_ylabel("Predicted")

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plot_filename = model_dir / f"{model.replace(' ', '_')}_station_{station}_h{step}.png"
                    fig.savefig(plot_filename)
                    plt.close(fig)

                print(f"[{variable}] Station {station}: Plots saved for {model}")

            # MAE vs Forecast Horizon
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in models:
                maes = []
                for step in steps:
                    subset = df_station[
                        (df_station['Model'] == model) &
                        (df_station['Step'] == step)
                    ]
                    if subset.empty:
                        maes.append(np.nan)
                        continue
                    mae = np.mean(np.abs(subset["True"] - subset["Predicted"]))
                    maes.append(mae)

                ax.plot(steps, maes, marker='o', label=model)

            ax.set_title(f"{variable} | Station {station} - MAE vs Forecast Horizon")
            ax.set_xlabel("Forecast Horizon (step)")
            ax.set_ylabel("Mean Absolute Error")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            fig.savefig(station_dir / f"mae_vs_horizon_station_{station}.png")
            plt.close(fig)

            print(f"[{variable}] Station {station}: MAE vs horizon saved")

if __name__ == "__main__":
    visualize_predictions(variables=["temperature_c"], per_station=True)
