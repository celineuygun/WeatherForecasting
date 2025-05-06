import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from models import get_models
from preprocess import preprocess_synop_data

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

def run_training(train_df, test_df, variable, station_id,
                 horizons, models, scale_needed,
                 all_metrics, all_predictions,
                 per_station=True,
                 var_thresh=0.01,
                 rf_n_estimators=100,
                 rf_max_depth=5,
                 rf_threshold='median'):

    drop_cols = ['datetime', variable]
    X_train_full = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test_full  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    for other_var in ['temperature_c', 'humidity', 'wind_speed']:
        if other_var != variable:
            cols_to_drop = [col for col in X_train_full.columns 
                            if col.startswith(f"{other_var}_lag_") or col.startswith(f"{other_var}_diff_")]
            X_train_full.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            X_test_full.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    y_train = train_df[[variable]].copy()
    y_test  = test_df[[variable]].copy()

    y_train_raw = pd.concat([
        y_train.shift(-h).rename(columns={variable: f"{variable}_t+{h}"}) for h in horizons
    ], axis=1).dropna()
    y_test_raw = pd.concat([
        y_test.shift(-h).rename(columns={variable: f"{variable}_t+{h}"}) for h in horizons
    ], axis=1).dropna()

    X_train = X_train_full.iloc[:len(y_train_raw)].reset_index(drop=True)
    X_test  = X_test_full.iloc[:len(y_test_raw)].reset_index(drop=True)
    y_train_raw = y_train_raw.reset_index(drop=True)
    y_test_raw  = y_test_raw.reset_index(drop=True)

    transformers = {}
    if variable in ['wind_speed']:
        transformers[variable] = PowerTransformer(method='yeo-johnson')
        y_train_arr = transformers[variable].fit_transform(y_train_raw.values)
        y_train_multi = pd.DataFrame(y_train_arr, columns=y_train_raw.columns)
    else:
        y_train_multi = y_train_raw.copy()

    vt = VarianceThreshold(threshold=var_thresh)
    X_tr_v = pd.DataFrame(vt.fit_transform(X_train),
                          columns=X_train.columns[vt.get_support()],
                          index=X_train.index)
    X_te_v = pd.DataFrame(vt.transform(X_test),
                          columns=X_tr_v.columns,
                          index=X_test.index)

    selector_rf = RandomForestRegressor(n_estimators=rf_n_estimators,
                                        max_depth=rf_max_depth,
                                        random_state=42,
                                        n_jobs=-1)
    selector_rf.fit(X_tr_v, y_train_multi)
    sfm = SelectFromModel(selector_rf, threshold=rf_threshold, prefit=True)
    support = sfm.get_support()

    selected_feats = list(set(X_tr_v.columns[support]) | set([col for col in X_tr_v.columns if col.startswith('station_')]))
    print(f"\n[FEATURE SELECTION] {station_id} — After SelectFromModel")
    print(f"  ➤ Selected features ({len(selected_feats)}): {selected_feats}")

    X_tr_sel = X_tr_v[selected_feats]
    X_te_sel = X_te_v[selected_feats]

    for model_name, base_model in models.items():
        print(f"\n  ➤ Training {model_name}")
        if (not per_station) or (model_name in scale_needed):
            scaler = MinMaxScaler()
            X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=selected_feats)
            X_te_scaled = pd.DataFrame(scaler.transform(X_te_sel), columns=selected_feats)
        else:
            X_tr_scaled, X_te_scaled = X_tr_sel.copy(), X_te_sel.copy()

        imp = SimpleImputer(strategy='mean')
        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr_scaled), columns=selected_feats)
        X_te_imp = pd.DataFrame(imp.transform(X_te_scaled), columns=selected_feats)

        reg = MultiOutputRegressor(base_model)
        reg.fit(X_tr_imp.values, y_train_multi.values)

        y_train_pred_raw = reg.predict(X_tr_imp.values)
        y_train_pred = (
            transformers[variable].inverse_transform(y_train_pred_raw)
            if variable in transformers else y_train_pred_raw
        )
        train_r2 = r2_score(y_train_raw.values, y_train_pred)

        y_pred_raw = reg.predict(X_te_imp.values)
        y_pred = (
            transformers[variable].inverse_transform(y_pred_raw)
            if variable in transformers else y_pred_raw
        )

        overall_r2 = r2_score(y_test_raw.values, y_pred)
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

def train_and_forecast(raw_csv_path,
                       forecast_horizon=6,
                       target_variables=None,
                       per_station=True):
    if target_variables is None:
        target_variables = ['temperature_c', 'humidity', 'wind_speed']
    horizons = list(range(1, forecast_horizon + 1))
    scale_needed = {'Linear Regression', 'SVM', 'KNN'}

    station_data = preprocess_synop_data(path=raw_csv_path,
                                         targets=target_variables,
                                         per_station=per_station)
    models = get_models()
    all_metrics = []

    results_dir = f"results/{'per_station' if per_station else 'merged'}"
    os.makedirs(results_dir, exist_ok=True)

    for var in target_variables:
        print(f"\n==================== {var.upper()} ====================\n")
        all_predictions = []
        if per_station:
            for sid, data in station_data.items():
                run_training(data['train_df'], data['test_df'],
                            var, sid,
                            horizons, models,
                            scale_needed,
                            all_metrics, all_predictions,
                            per_station)
        else:
            run_training(station_data['train_df'], station_data['test_df'],
                        var, "merged",
                        horizons, models,
                        scale_needed,
                        all_metrics, all_predictions,
                        per_station)

        pd.DataFrame(all_predictions).to_csv(
            os.path.join(results_dir, f"predictions_{var}.csv"), index=False)
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(results_dir, "metrics.csv"), index=False)
    print("\nINFO: Training and forecasting completed.")

if __name__ == '__main__':
    train_and_forecast(
        raw_csv_path='dataset/synop.csv',
        forecast_horizon=6,
        target_variables=['temperature_c', 'humidity', 'wind_speed'],
        per_station=True
    )