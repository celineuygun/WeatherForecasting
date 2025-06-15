import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from pathlib import Path
import numpy as np
import warnings

sns.set(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning)

def visualize_predictions(
    variables=["temperature_c", "humidity", "wind_speed"],
    per_station=True,
    allowed_steps=[3, 6, 9]
):
    """
    Generates and saves visualization plots for model predictions, including:

    - Time series comparison (True vs Predicted)
    - Rolling MAE
    - Residual distributions
    - True vs Predicted scatter with R²
    - MAE vs Horizon for each model

    Args:
        variables (list[str]): List of variable names to visualize.
        per_station (bool): If True, visualize each station separately; otherwise, merged results.
        allowed_steps (list[int]): Forecast steps (in hours) to include in the visualizations.

    Saves:
        PNG plots to results/per_station/plots/ or results/merged/plots/ directory structure.
    """

    base_dir = "results/per_station" if per_station else "results/merged"

    for variable in variables:
        # Load prediction CSV for the variable
        predictions_file = f"{base_dir}/predictions_{variable}.csv"
        if not os.path.exists(predictions_file):
            print(f"ERROR: File not found: {predictions_file}")
            continue

        # Load predictions into DataFrame
        print(f"\nINFO: Visualizing: {predictions_file}")
        preds_df = pd.read_csv(predictions_file)

        # Add missing 'Station' column if working with merged data
        if "Station" not in preds_df.columns:
            preds_df["Station"] = "merged"

        # Keep only specified forecast steps
        preds_df = preds_df[preds_df["Step"].isin(allowed_steps)]

        # Set output directory for plots
        plot_base_dir = Path(base_dir) / "plots" / variable
        plot_base_dir.mkdir(parents=True, exist_ok=True)

        # Extract unique station and model names
        stations = preds_df["Station"].unique()
        models   = preds_df["Model"].unique()
        steps    = allowed_steps

        # Determine global maximum y-density to normalize residual plots
        global_density_max = 0
        for station in stations:
            for model in models:
                for step in steps:
                    subset = preds_df[
                        (preds_df["Station"] == station) &
                        (preds_df["Model"]   == model)   &
                        (preds_df["Step"]    == step)
                    ]
                    if subset.empty:
                        continue

                    # Plot histogram of residuals (True - Predicted) to get KDE line
                    h = sns.histplot(
                        subset["Predicted"] - subset["True"],
                        bins=30, stat="density", kde=True
                    )

                    lines = h.get_lines()
                    if lines:
                        y = lines[0].get_data()[1]
                        global_density_max = max(global_density_max, np.nanmax(y))

                    plt.clf() # Clear plot for next loop

        # Main loop for plotting visualizations
        for station in stations:
            station_dir = plot_base_dir / f"station_{station}"
            station_dir.mkdir(exist_ok=True)

            # Filter predictions for current station
            df_st = preds_df[preds_df["Station"] == station].copy()

            # Compute residuals
            df_st["residual"] = df_st["Predicted"] - df_st["True"]

            # Determine axis limits
            val_min = df_st[["True","Predicted"]].min().min()
            val_max = df_st[["True","Predicted"]].max().max()
            res_min = df_st["residual"].min()
            res_max = df_st["residual"].max()

            for model in models:
                model_dir = station_dir / model.replace(" ", "_")
                model_dir.mkdir(exist_ok=True)

                for step in steps:
                    # Filter for this model and horizon step
                    subset = df_st[
                        (df_st["Model"] == model) &
                        (df_st["Step"]  == step)
                    ].reset_index(drop=True)
                    if subset.empty:
                        continue

                    # rolling MAE
                    rolling_mae = np.abs(subset["residual"])\
                                   .rolling(window=100, min_periods=1).mean()
                    mae_max = rolling_mae.max()

                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    fig.suptitle(f"{variable} | Station {station} – {model} (t+{step})", fontsize=14)

                    # 1) True vs Pred
                    axs[0,0].plot(subset["True"], label="True", alpha=0.7)
                    axs[0,0].plot(subset["Predicted"], label="Pred", alpha=0.7)
                    axs[0,0].legend(); axs[0,0].set_title("True vs Pred")
                    axs[0,0].set_ylabel(variable)

                    # 2) Rolling MAE
                    axs[0,1].plot(rolling_mae)
                    axs[0,1].set_ylim(0, mae_max*1.1)
                    axs[0,1].set_title("Rolling MAE (win=100)")

                    # 3) Residual histogram
                    sns.histplot(
                        subset["residual"], bins=30, stat="density", kde=True,
                        ax=axs[1,0], color="salmon"
                    )
                    axs[1,0].set_xlim(res_min, res_max)
                    axs[1,0].set_ylim(0, global_density_max*1.1)
                    axs[1,0].set_title("Residuals")

                    # 4) Scatter True vs Pred
                    r2 = r2_score(subset["True"], subset["Predicted"])
                    sns.scatterplot(
                        x="True", y="Predicted", data=subset,
                        alpha=0.3, ax=axs[1,1], color="crimson"
                    )
                    axs[1,1].plot([val_min, val_max],[val_min, val_max], "--", color="black")
                    axs[1,1].set_xlim(val_min, val_max)
                    axs[1,1].set_ylim(val_min, val_max)
                    axs[1,1].text(val_min, val_max, f"R²={r2:.2f}", fontsize=10)
                    axs[1,1].set_title("Scatter")

                    plt.tight_layout(rect=[0,0.03,1,0.95])
                    fn = model_dir / f"{model.replace(' ','_')}_h{step}.png"
                    fig.savefig(fn)
                    plt.close(fig)

                print(f"[{variable}] Station {station}: Plots saved for {model}")

            # MAE vs Horizon
            fig, ax = plt.subplots(figsize=(8,4))
            for model in models:
                maes = []
                for step in steps:
                    sub = df_st[
                        (df_st["Model"] == model) &
                        (df_st["Step"]  == step)
                    ]
                    maes.append(sub["True"].subtract(sub["Predicted"]).abs().mean()
                               if not sub.empty else np.nan)
                ax.plot(steps, maes, marker="o", label=model)

            ax.set_title(f"{variable} | Station {station} – MAE vs Horizon")
            ax.set_xlabel("Forecast Horizon (hours ahead)")
            ax.set_ylabel("MAE")
            ax.set_xticks(steps)
            ax.legend()
            plt.tight_layout()
            fig.savefig(station_dir / f"mae_vs_horizon_station_{station}.png")
            plt.close(fig)

            print(f"[{variable}] Station {station}: MAE vs horizon saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize prediction results.")
    parser.add_argument("--per_station", action="store_true", help="Visualize per station instead of merged")

    args = parser.parse_args()

    visualize_predictions(
        variables=["temperature_c"],
        per_station=args.per_station
    )