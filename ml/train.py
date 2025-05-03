import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from models import get_models
from preprocess import preprocess_synop_data

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((y_true - y_pred) / y_true)
        mape[~np.isfinite(mape)] = 0
    return np.mean(mape) * 100

def create_multi_horizon_targets(df, target_col, horizons):
    return pd.concat([
        df[target_col].shift(-h).rename(f"{target_col}_t+{h}") for h in horizons
    ], axis=1).dropna()

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, rmse, r2, evs, mape

def train_and_forecast(raw_csv_path, forecast_horizon=6, target_variables=["temperature_c", "humidity", "wind_speed"]):
    os.makedirs("results", exist_ok=True)
    horizons = list(range(1, forecast_horizon + 1))
    scale_needed = {'Linear Regression', 'SVM', 'KNN'}
    station_data = preprocess_synop_data(raw_csv_path, target_variables=target_variables)
    models = get_models()

    for variable in target_variables:
        print(f"\n==================== {variable.upper()} ====================\n")
        all_metrics, all_predictions = [], []

        for station_id, data in station_data.items():
            print(f"\nINFO: Training on station {station_id} — variable: {variable}")

            df_train = data['train_df'].copy()
            df_test = data['test_df'].copy()

            y_train = df_train[variable]
            y_test = df_test[variable]

            X_train = df_train.drop(columns=['datetime'] + target_variables)
            X_test = df_test.drop(columns=['datetime'] + target_variables)

            df_train = X_train.copy()
            df_test = X_test.copy()
            df_train['target'] = y_train
            df_test['target'] = y_test

            df_train.dropna(inplace=True)
            df_test.dropna(inplace=True)

            y_train_multi = create_multi_horizon_targets(df_train, 'target', horizons)
            y_test_multi = create_multi_horizon_targets(df_test, 'target', horizons)
            X_train = df_train.iloc[:len(y_train_multi)].drop(columns=['target'])
            X_test = df_test.iloc[:len(y_test_multi)].drop(columns=['target'])

            for model_name, base_model in models.items():
                print(f"  ➤ {model_name}")

                if model_name in scale_needed:
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test

                model = MultiOutputRegressor(base_model)
                model.fit(X_train_scaled, y_train_multi)
                y_pred = model.predict(X_test_scaled)
                y_train_pred = model.predict(X_train_scaled)

                train_r2 = r2_score(y_train_multi, y_train_pred)
                test_r2 = r2_score(y_test_multi, y_pred)

                for i, step in enumerate(horizons):
                    yt = y_test_multi.iloc[:, i]
                    yp = y_pred[:, i]
                    mae, mse, rmse, r2s, evs, mape = evaluate_forecast(yt, yp)

                    all_metrics.append({
                        'Variable': variable, 'Station': station_id,
                        'Model': model_name, 'Step': step,
                        'Train R2': train_r2, 'Test R2': test_r2,
                        'MAE': mae, 'MSE': mse,
                        'RMSE': rmse, 'R2 Step': r2s,
                        'EVS': evs, 'MAPE': mape
                    })

                    all_predictions.extend([
                        {'Variable': variable, 'Station': station_id, 'Model': model_name,
                         'Step': step, 'True': t, 'Predicted': p}
                        for t, p in zip(yt.values, yp)
                    ])

        metrics_df = pd.DataFrame(all_metrics)
        preds_df = pd.DataFrame(all_predictions)

        metrics_path = f"results/metrics_{variable}.csv"
        preds_path = f"results/predictions_{variable}.csv"

        metrics_df.to_csv(metrics_path, index=False)
        preds_df.to_csv(preds_path, index=False)

        print(f"\nINFO: Saved {metrics_path} and {preds_path}")

if __name__ == "__main__":
    train_and_forecast(
        raw_csv_path="dataset/synop_data.csv",
        forecast_horizon=6,
        target_variables=["temperature_c", "humidity", "wind_speed"]
    )
