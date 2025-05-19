import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from models import get_models
from preprocess import preprocess_synop_data
from utils import evaluate, compare_models_ttest, prepare_target_shifted, mutual_info_feature_selection

def run_training(train_df, test_df, variable, station_id,
                 horizons, models, scale_needed,
                 all_metrics, all_predictions,
                 selected_features):

    drop_cols = ['datetime', variable]
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    
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
            mae, mse, rmse, r2s, evs, mape = evaluate(yt, yp)
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
            X = X.select_dtypes(include=[np.number])

            X = X.dropna(axis=0)
            y = y.loc[X.index]

            selected_features = mutual_info_feature_selection(
                X, y[var], top_k=20, min_mi=0.01, verbose=True
            )

            station_cols = [col for col in X.columns if col.startswith("station_") and col[8:].isdigit()]
            if station_cols:
                print(f"[FORCE INCLUDE] Adding station columns: {station_cols}")
                selected_features = list(set(selected_features + station_cols))



            run_training(train_df, test_df, var, "merged",
                         horizons, models, scale_needed,
                         all_metrics, all_predictions, selected_features)

        pd.DataFrame(all_predictions).to_csv(os.path.join(results_dir, f"predictions_{var}.csv"), index=False)

    pd.DataFrame(all_metrics).to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print("\nINFO: Training and forecasting completed.")


if __name__ == '__main__':
    forecast_horizon = 6
    per_station = False
    train_and_forecast(
        raw_csv_path='dataset/synop.csv',
        forecast_horizon=forecast_horizon,
        target_variables=['temperature_c'],
        per_station=per_station
    )

    results_dir = "results/per_station" if per_station else "results/merged"
    metrics_df = pd.read_csv(os.path.join(results_dir, "metrics.csv"))

    if per_station:
        for var in ['temperature_c']:
            for h in range(1, forecast_horizon + 1):
                compare_models_ttest(metrics_df, var, h)

