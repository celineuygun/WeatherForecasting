import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ttest_rel
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

HORIZONS = [3, 6, 9] # Forecast horizons in hours

# Abbreviations for model names used in display and comparisons
MODEL_ABBR = {
    "Linear Regression": "LR",
    "Random Forest": "RF",
    "Gradient Boosting": "GB",
    "Neural Network": "NN",
    "XGBoost": "XGB"
}

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between predicted and actual values.

    Args:
        y_true (array-like): Ground truth (actual) target values.
        y_pred (array-like): Forecasted or predicted values.

    Returns:
        float: MAPE value expressed as a percentage.
    """

    # Convert inputs to NumPy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle division by zero or invalid cases gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_true - y_pred) / y_true)
        mape[~np.isfinite(mape)] = 0 # Set infinities or NaNs to 0

    return np.mean(mape) * 100 # Return MAPE as percentage

def evaluate(y_true, y_pred):
    """
    Computes a suite of regression evaluation metrics for model performance.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values from the model.

    Returns:
        tuple: (MAE, MSE, RMSE, R², Explained Variance, MAPE)
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, r2, evs, mape

def compare_models_ttest(metrics_df, variable, horizon):
    """
    Performs paired t-tests between all model combinations for a specific variable and forecast horizon.

    Args:
        metrics_df (DataFrame): DataFrame containing evaluation metrics per model and step.
        variable (str): Name of the variable being forecasted.
        horizon (int): Forecast horizon (in hours) to compare.

    Prints:
        - p-values of paired t-tests for each metric and model pair.
        - Asterisks (*) next to statistically significant results (p < 0.05).
    """

    # Filter relevant subset for selected variable and horizon
    subset = metrics_df[
        (metrics_df['Variable'] == variable) &
        (metrics_df['Step'] == horizon)
    ]
    models = sorted(subset['Model'].unique()) # Get sorted list of models
    metrics_to_test = ['MAE', 'RMSE', 'R2 Step'] # Metrics to compare

    print(f"\n{variable} @ t+{horizon}")
    print(f"{'Metric':<12} {'Models':<12} {'p-val':<6} {'*'}")
    print('-' * 35)

    # Iterate over all unique model combinations
    for metric in metrics_to_test:
        for m1, m2 in combinations(models, 2):
            a = subset[subset['Model'] == m1][metric].values
            b = subset[subset['Model'] == m2][metric].values

            if len(a) != len(b):
                continue  # Skip if unequal sample sizes

            # Perform paired t-test
            t_stat, p_value = ttest_rel(a, b, nan_policy='omit')
            signif = '*' if p_value < 0.05 else ''

            # Get abbreviations for model names
            m1_abbr = MODEL_ABBR.get(m1, m1)
            m2_abbr = MODEL_ABBR.get(m2, m2)
            combo = f"{m1_abbr} vs {m2_abbr}"

            # Display result row
            print(f"{metric:<12} {combo:<12} {p_value:<6.3f} {signif}")

def plot_mutual_info(X, y, top_n=30, min_mi=0.1, per_station=True):
    """
    Calculates mutual information (MI) between features and target, plots the top scores,
    and saves the visualization as a PNG.

    Args:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
        top_n (int): Maximum number of top features to visualize.
        min_mi (float): Minimum MI score threshold to include a feature.
        per_station (bool): Determines output directory (per_station or merged).

    Returns:
        Series: All mutual information scores (not just the plotted subset).
    """
    # Compute mutual information scores
    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns)

    # Clean and filter MI scores
    mi_series = mi_series.dropna()
    mi_series = mi_series[mi_series.index.notnull()]
    mi_series = mi_series[mi_series >= min_mi].sort_values()

    top_features = mi_series[-top_n:] # Select top features

    base = 'per_station' if per_station else 'merged'
    save_dir = os.path.join('results', base, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    # Plot MI scores as horizontal bar chart
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, max(4, len(top_features) * 0.3)))
    sns.barplot(x=top_features.values, y=top_features.index, orient='h', color='skyblue')
    plt.xlabel("Mutual Information Score")
    plt.title(f"Top {len(top_features)} Features by MI with {y.name}")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot
    fname = f"mi_{y.name}.png" if y.name else "mutual_info.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"INFO: Saved mutual info plot to {path}")

    return mi_series

def mutual_info_feature_selection(X, y, top_k=None, min_mi=None, per_station=True, verbose=True):
    """
    Selects the most relevant features based on mutual information scores.

    Args:
        X (DataFrame): Input features.
        y (Series): Target variable.
        top_k (int, optional): Select the top K features.
        min_mi (float, optional): Minimum mutual information threshold.
        per_station (bool): For visualization purposes (used in plotting).
        verbose (bool): Whether to print selected feature names.

    Returns:
        list: Names of selected features.
    """

    # Generate mutual information plot (for visualization)
    plot_mutual_info(X, y, top_n=top_k, min_mi=min_mi, per_station=per_station)

    # Compute mutual information scores
    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # Apply minimum MI threshold if specified
    if min_mi is not None:
        mi_series = mi_series[mi_series >= min_mi]

    # Select top K features if specified
    if top_k is not None:
        mi_series = mi_series.head(top_k)

    selected = mi_series.index.tolist() # Extract selected feature names

    if verbose:
        print(f"\n[MI SELECTION] {len(selected)} features selected (top_k={top_k}, min_mi={min_mi}): {selected}")

    return selected 

def prepare_target_shifted(y, variable, horizons):
    """
    Generates multi-output target columns by shifting the original target variable
    for each forecast horizon.

    Args:
        y (DataFrame): Single-column target variable data.
        variable (str): Name of the target variable.
        horizons (list[int]): Forecast horizons (in hours) to shift by.

    Returns:
        tuple: (shifted_targets, shifted_targets) — both are the same DataFrame.
    """
    
    # Create multi-step shifted target columns for each horizon
    y_raw = pd.concat([
        y.shift(-h).rename(columns={variable: f"{variable}_t+{h}"}) for h in horizons
    ], axis=1).dropna()

    return y_raw, y_raw # Return target matrix twice (multi-output and raw format)