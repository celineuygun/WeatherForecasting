import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.base import clone
from collections import Counter
from scipy.stats import ttest_rel
from models import get_models
from preprocess import preprocess_synop_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_true - y_pred) / y_true)
        mape[~np.isfinite(mape)] = 0
    return np.mean(mape) * 100

def evaluate_forecast(y_true, y_pred):
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
    """
    Daha temiz ve sıralı MI grafiği çizer.
    
    Params:
    - X: pd.DataFrame
    - y: pd.Series
    - top_n: en çok gösterilecek feature sayısı
    - min_mi: minimum MI değeri filtresi

    Returns:
    - pd.Series of MI scores (tümü)
    """
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
    """
    Flexible mutual information-based feature selection.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series, target variable
    - top_k: int or None, max number of top features to keep
    - min_mi: float or None, minimum mutual information threshold
    - verbose: bool, print selected features

    Returns:
    - List of selected feature names
    """
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

def run_training(train_df, test_df, variable, station_id,
                 horizons, models, scale_needed,
                 all_metrics, all_predictions,
                 selected_features):

    drop_cols = ['datetime', variable]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    for other_var in ['temperature_c', 'humidity', 'wind_speed']:
        if other_var != variable:
            X_train = X_train.drop(columns=[col for col in X_train.columns if col.startswith(f"{other_var}_lag_") or col.startswith(f"{other_var}_diff_")], errors='ignore')
            X_test = X_test.drop(columns=[col for col in X_test.columns if col.startswith(f"{other_var}_lag_") or col.startswith(f"{other_var}_diff_")], errors='ignore')

    y_train_multi, y_train_raw = prepare_target_shifted(train_df[[variable]], variable, horizons)
    y_test_raw, _ = prepare_target_shifted(test_df[[variable]], variable, horizons)

    X_train = X_train.iloc[:len(y_train_multi)][selected_features].reset_index(drop=True)
    X_test = X_test.iloc[:len(y_test_raw)][selected_features].reset_index(drop=True)

    for model_name, base_model in models.items():
        print(f"  ➤ Training {model_name}")
        if model_name in scale_needed:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

        imputer = SimpleImputer(strategy='mean')
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=selected_features)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=selected_features)

        reg = MultiOutputRegressor(base_model)
        reg.fit(X_train_imp, y_train_multi)

        y_pred = reg.predict(X_test_imp)
        train_pred = reg.predict(X_train_imp)

        overall_r2 = r2_score(y_test_raw.values, y_pred)
        train_r2 = r2_score(y_train_raw.values, train_pred)

        for i, h in enumerate(horizons):
            yt = y_test_raw.iloc[:, i]
            yp = y_pred[:, i]
            mae, mse, rmse, r2s, evs, mape = evaluate_forecast(yt, yp)
            all_metrics.append({
                'Variable': variable,
                'Station': station_id,
                'Model': model_name,
                'Step': h,
                'Train R2': train_r2,
                'Test R2': overall_r2,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2 Step': r2s,
                'EVS': evs,
                'MAPE': mape
            })
            all_predictions.extend([
                {
                    'Variable': variable,
                    'Station': station_id,
                    'Model': model_name,
                    'Step': h,
                    'True': true,
                    'Predicted': pred
                } for true, pred in zip(yt, yp)
            ])

def train_and_forecast(raw_csv_path, forecast_horizon=6, target_variables=None, per_station=True):
    if target_variables is None:
        target_variables = ['temperature_c', 'humidity', 'wind_speed']
    horizons = list(range(1, forecast_horizon + 1))
    scale_needed = {'Linear Regression', 'Neural Network'}

    station_data = preprocess_synop_data(path=raw_csv_path, targets=target_variables, per_station=per_station)
    models = get_models()
    all_metrics = []

    results_dir = f"results/{'per_station' if per_station else 'merged'}"
    os.makedirs(results_dir, exist_ok=True)

    for var in target_variables:
        print(f"\n==================== {var.upper()} ====================\n")
        all_predictions = []

        if per_station:
            all_train_dfs = []
            for sid, data in station_data.items():
                df = data['train_df'].copy()
                df["station_id"] = sid
                all_train_dfs.append(df)
            merged_train = pd.concat(all_train_dfs).reset_index(drop=True)

            X_all = merged_train.drop(columns=['datetime', var], errors='ignore')
            for other_var in ['temperature_c', 'humidity', 'wind_speed']:
                if other_var != var:
                    X_all = X_all.drop(columns=[col for col in X_all.columns if col.startswith(f"{other_var}_lag_") or col.startswith(f"{other_var}_diff_")], errors='ignore')
            y_all = merged_train[[var]].dropna()
            X_all = X_all.loc[y_all.index]

            selected_features = mutual_info_feature_selection(
                X_all, y_all[var], top_k=20, min_mi=0.01, verbose=True
            )

            for sid, data in station_data.items():
                print(f"\n[Station {sid}]")
                train_df = data['train_df']
                test_df = data['test_df']

                run_training(train_df, test_df, var, sid,
                             horizons, models, scale_needed,
                             all_metrics, all_predictions, selected_features)
        else:
            data = station_data
            train_df = data['train_df']
            test_df = data['test_df']

            print(f"\n[Global Training]")
            X = train_df.drop(columns=['datetime', var], errors='ignore')
            for other_var in ['temperature_c', 'humidity', 'wind_speed']:
                if other_var != var:
                    X = X.drop(columns=[col for col in X.columns if col.startswith(f"{other_var}_lag_") or col.startswith(f"{other_var}_diff_")], errors='ignore')
            y = train_df[[var]].dropna()
            X = X.loc[y.index]

            selected_features = mutual_info_feature_selection(
                X, y[var], top_k=20, min_mi=0.01, verbose=True
            )

            run_training(train_df, test_df, var, "merged",
                         horizons, models, scale_needed,
                         all_metrics, all_predictions, selected_features)

        pd.DataFrame(all_predictions).to_csv(os.path.join(results_dir, f"predictions_{var}.csv"), index=False)

    pd.DataFrame(all_metrics).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("\nINFO: Training and forecasting completed.")


if __name__ == '__main__':
    forecast_horizon = 6
    train_and_forecast(
        raw_csv_path='dataset/synop.csv',
        forecast_horizon=forecast_horizon,
        target_variables=['temperature_c'],
        per_station=True
    )

    metrics_df = pd.read_csv("results/per_station/metrics.csv")
    for var in ['temperature_c']:
        for h in range(1, forecast_horizon + 1):
            compare_models_ttest(metrics_df, var, h)

