import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

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
    print(f"\n[STAT TEST] Variable: {variable}, Horizon: t+{horizon}")
    subset = metrics_df[(metrics_df['Variable'] == variable) & (metrics_df['Step'] == horizon)]
    models = subset['Model'].unique()
    metrics_to_test = ['MAE', 'RMSE', 'R2 Step']

    for metric in metrics_to_test:
        print(f"\nMetric: {metric}")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]

                scores_a = subset[subset['Model'] == model_a][metric].values
                scores_b = subset[subset['Model'] == model_b][metric].values

                if len(scores_a) != len(scores_b):
                    print(f"  ✗ Skipping {model_a} vs {model_b} (unequal sample sizes)")
                    continue

                t_stat, p_value = ttest_rel(scores_a, scores_b)
                print(f"  ➤ {model_a} vs {model_b}: t={t_stat:.3f}, p={p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

def plot_mutual_info(X, y, top_n=30, min_mi=0.0):
    mi = mutual_info_regression(X, y, discrete_features='auto')

    mi_series = pd.Series(mi, index=X.columns)
    mi_series = mi_series.dropna()
    mi_series = mi_series[mi_series.index.notnull()]
    mi_series = mi_series[mi_series >= min_mi].sort_values()

    top_features = mi_series[-top_n:]

    plt.figure(figsize=(8, max(4, len(top_features) * 0.3)))
    sns.barplot(x=top_features.values, y=top_features.index, orient='h', color='skyblue')
    plt.xlabel("Mutual Information Score")
    plt.title(f"Top {len(top_features)} Features by MI with Target")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return mi_series

def mutual_info_feature_selection(X, y, top_k=None, min_mi=None, verbose=True):
    # plot_mutual_info(X, y, top_n=top_k, min_mi=min_mi)

    mi = mutual_info_regression(X, y, discrete_features='auto')
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    if min_mi is not None:
        mi_series = mi_series[mi_series >= min_mi]

    if top_k is not None:
        mi_series = mi_series.head(top_k)

    selected = mi_series.index.tolist()

    if verbose:
        print(f"[MI SELECTION] {len(selected)} features selected (top_k={top_k}, min_mi={min_mi}): {selected}")

    return selected

def prepare_target_shifted(y, variable, horizons):
    y_raw = pd.concat([
        y.shift(-h).rename(columns={variable: f"{variable}_t+{h}"}) for h in horizons
    ], axis=1).dropna()
    return y_raw, y_raw