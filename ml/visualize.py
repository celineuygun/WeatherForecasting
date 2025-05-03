import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from pathlib import Path
import numpy as np

sns.set(style="whitegrid")

def visualize_predictions(predictions_file="results/predictions.csv"):
    preds_df = pd.read_csv(predictions_file)

    base_dir = Path("results/plots")
    base_dir.mkdir(parents=True, exist_ok=True)

    stations = preds_df['Station'].unique()
    models = preds_df['Model'].unique()
    steps = sorted(preds_df['Step'].unique())

    for station in stations:
        station_dir = base_dir / f"station_{station}"
        station_dir.mkdir(exist_ok=True)

        # Precompute global axis limits per station
        df_station = preds_df[preds_df['Station'] == station].copy()
        df_station["residual"] = df_station["Predicted"] - df_station["True"]

        residual_min, residual_max = df_station["residual"].min(), df_station["residual"].max()
        temp_min, temp_max = df_station[["True", "Predicted"]].min().min(), df_station[["True", "Predicted"]].max().max()
        mae_max = df_station["residual"].abs().rolling(window=100, min_periods=1).mean().max()

        for model in models:
            model_dir = station_dir / model.replace(' ', '_')
            model_dir.mkdir(exist_ok=True)

            for step in steps:
                subset = df_station[
                    (df_station['Model'] == model) &
                    (df_station['Step'] == step)
                ]

                if subset.empty:
                    continue

                fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f"Station {station} - {model} (t+{step})", fontsize=14)

                # Prediction vs True
                axs[0, 0].plot(subset['True'].values, label='True', alpha=0.7)
                axs[0, 0].plot(subset['Predicted'].values, label='Predicted', alpha=0.7)
                axs[0, 0].set_ylabel("Temperature (°C)")
                axs[0, 0].set_ylim(temp_min, temp_max)
                axs[0, 0].legend()
                axs[0, 0].set_title("Prediction vs. True")

                # Rolling MAE
                rolling_mae = np.abs(subset["residual"]).rolling(window=100, min_periods=1).mean()
                axs[0, 1].plot(rolling_mae)
                axs[0, 1].set_ylim(0, mae_max * 1.05)
                axs[0, 1].set_title("Rolling MAE (window=100)")
                axs[0, 1].set_ylabel("MAE")

                # Residuals Histogram
                sns.histplot(subset["residual"], bins=30, kde=True, ax=axs[1, 0], color="salmon", stat="density")
                axs[1, 0].set_xlim(residual_min, residual_max)
                axs[1, 0].set_ylim(0, 0.25)
                axs[1, 0].set_xlabel("Predicted - True")
                axs[1, 0].set_title("Residuals Histogram")

                # True vs Predicted Scatter
                r_val = r2_score(subset['True'], subset['Predicted'])
                sns.scatterplot(x="True", y="Predicted", data=subset, alpha=0.3, ax=axs[1, 1], color="crimson")
                axs[1, 1].plot([temp_min, temp_max], [temp_min, temp_max], linestyle="--", color="black")
                axs[1, 1].set_xlim(temp_min, temp_max)
                axs[1, 1].set_ylim(temp_min, temp_max)
                axs[1, 1].text(temp_min + 1, temp_max - 2, f"R² = {r_val:.2f}", fontsize=10)
                axs[1, 1].set_title("True vs Predicted Scatter")
                axs[1, 1].set_xlabel("True")
                axs[1, 1].set_ylabel("Predicted")

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = model_dir / f"{model.replace(' ', '_')}_station_{station}_h{step}.png"
                fig.savefig(plot_filename)
                plt.close(fig)

            print(f"[Station {station}] Plots saved for {model}")

        for step in steps:
            fig, ax = plt.subplots(figsize=(10, 5))

            for model in models:
                subset = df_station[df_station["Model"] == model]
                if subset.empty:
                    continue

                sns.histplot(
                    subset["residual"],
                    bins=40,
                    kde=True,
                    stat="density",
                    label=model,
                    element="step",
                    fill=True,
                    ax=ax,
                    alpha=0.5
                )

            ax.axvline(0, linestyle='--', color='black', linewidth=1)
            ax.set_title(f"Station {station} - Residual Histogram")
            ax.set_xlabel("Prediction Residual (y_pred - y_true)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            hist_compare_path = station_dir / f"residual_histogram_all_models_station_{station}.png"
            plt.savefig(hist_compare_path)
            plt.close(fig)

            # MAE vs Forecast Horizon
            fig, ax = plt.subplots(figsize=(10, 5))

            for model in models:
                model_maes = []
                for step in steps:
                    subset = df_station[
                        (df_station['Model'] == model) &
                        (df_station['Step'] == step)
                    ]
                    if subset.empty:
                        model_maes.append(np.nan)
                        continue

                    mae = np.mean(np.abs(subset["True"] - subset["Predicted"]))
                    model_maes.append(mae)

                ax.plot([int(s) for s in steps], model_maes, marker='o', label=model)

            ax.set_xticks([int(s) for s in steps])
            ax.set_title(f"Station {station} - MAE vs Forecast Horizon")
            ax.set_xlabel("Forecast Horizon (step)")
            ax.set_ylabel("Mean Absolute Error (MAE)")
            ax.legend()
            ax.grid(True)

            plt.tight_layout()
            mae_cmp_path = station_dir / f"model_comparison_mae_vs_horizon_station_{station}.png"
            plt.savefig(mae_cmp_path)
            plt.close(fig)

        print(f"INFO: Comparison plots saved to {station_dir}\n")

if __name__ == "__main__":
    visualize_predictions()
