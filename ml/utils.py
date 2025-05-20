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

model_abbr = {
    "Linear Regression": "LR",
    "Random Forest": "RF",
    "Gradient Boosting": "GB",
    "Neural Network": "NN",
    "XGBoost": "XGB"
}

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_true - y_pred) / y_true)
        mape[~np.isfinite(mape)] = 0
    return np.mean(mape) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, r2, evs, mape

def compare_models_ttest(metrics_df, variable, horizon):
    """
    Perform paired t-test between each pair of models for a given variable and horizon.
    Outputs a concise comparison table: one row per metric & model-pair.
    """
    subset = metrics_df[
        (metrics_df['Variable'] == variable) &
        (metrics_df['Step'] == horizon)
    ]
    models = sorted(subset['Model'].unique())
    metrics_to_test = ['MAE', 'RMSE', 'R2 Step']

    print(f"\n{variable} @ t+{horizon}")
    print(f"{'Metric':<12} {'Models':<12} {'p-val':<6} {'*'}")
    print('-' * 35)

    for metric in metrics_to_test:
        for m1, m2 in combinations(models, 2):
            a = subset[subset['Model'] == m1][metric].values
            b = subset[subset['Model'] == m2][metric].values
            if len(a) != len(b):
                continue
            t_stat, p_value = ttest_rel(a, b, nan_policy='omit')
            signif = '*' if p_value < 0.05 else ''
            m1_abbr = model_abbr.get(m1, m1)
            m2_abbr = model_abbr.get(m2, m2)
            combo = f"{m1_abbr} vs {m2_abbr}"
            print(f"{metric:<12} {combo:<12} {p_value:<6.3f} {signif}")

def plot_mutual_info(X, y, top_n=30, min_mi=0.1, per_station=True):
    """
    Compute and plot mutual information scores between features X and target y,
    then save the bar chart automatically under results/per_station or results/merged.

    Args:
      X: DataFrame of features
      y: Series of target values (y.name used for filename)
      top_n: number of top features to display
      min_mi: minimum mutual information threshold to include
      per_station: if True, save under results/per_station, else results/merged

    Returns:
      mi_series: pd.Series of all MI scores
    """
    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns)
    mi_series = mi_series.dropna()
    mi_series = mi_series[mi_series.index.notnull()]
    mi_series = mi_series[mi_series >= min_mi].sort_values()

    top_features = mi_series[-top_n:]

    base = 'per_station' if per_station else 'merged'
    save_dir = os.path.join('results', base, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, max(4, len(top_features) * 0.3)))
    sns.barplot(x=top_features.values, y=top_features.index, orient='h', color='skyblue')
    plt.xlabel("Mutual Information Score")
    plt.title(f"Top {len(top_features)} Features by MI with {y.name}")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    fname = f"mi_{y.name}.png" if y.name else "mutual_info.png"
    path = os.path.join(save_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"INFO: Saved mutual info plot to {path}")

    return mi_series

def mutual_info_feature_selection(X, y, top_k=None, min_mi=None, per_station=True, verbose=True):
    plot_mutual_info(X, y, top_n=top_k, min_mi=min_mi, per_station=per_station)

    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    if min_mi is not None:
        mi_series = mi_series[mi_series >= min_mi]

    if top_k is not None:
        mi_series = mi_series.head(top_k)

    selected = mi_series.index.tolist()

    if verbose:
        print(f"\n[MI SELECTION] {len(selected)} features selected (top_k={top_k}, min_mi={min_mi}): {selected}")

    return selected

def prepare_target_shifted(y, variable, horizons):
    y_raw = pd.concat([
        y.shift(-h).rename(columns={variable: f"{variable}_t+{h}"}) for h in horizons
    ], axis=1).dropna()
    return y_raw, y_raw