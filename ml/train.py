import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import joblib
from models import get_models
from preprocess import preprocess_synop_data
from utils import HORIZONS, evaluate, prepare_target_shifted, mutual_info_feature_selection

def run_training(train_df, test_df, variable, station_id,
                 horizons, models, scale_needed,
                 all_metrics, all_predictions,
                 selected_features):
    """
    Trains multiple machine learning models for a specific variable and station, 
    performs multi-step forecasting, and evaluates performance.

    Args:
        train_df (DataFrame): Preprocessed training dataset.
        test_df (DataFrame): Preprocessed testing dataset.
        variable (str): Target variable to forecast (e.g., 'temperature_c').
        station_id (str): Identifier for the station ('merged' for global).
        horizons (list[int]): Forecast horizons in hours (e.g., [3, 6, 9]).
        models (dict): Dictionary mapping model names to sklearn-compatible estimators.
        scale_needed (set): Model names that require feature scaling.
        all_metrics (list): List to collect performance metrics for each step.
        all_predictions (list): List to collect forecasted values for saving/export.
        selected_features (list): List of selected input feature column names.

    Saves:
        - Trained model to disk using joblib.
        - Evaluation metrics appended to all_metrics.
        - Forecast results appended to all_predictions.
    """

    # Drop rows with missing values and reset index
    drop_cols = ['datetime', variable]
    train_df = train_df.dropna().reset_index(drop=True)
    test_df  = test_df.dropna().reset_index(drop=True)

    # Drop datetime and target variable columns from features
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test  = test_df .drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Remove lag and diff features of other variables
    for other in ['temperature_c','humidity','wind_speed']:
        if other != variable:
            bad = [c for c in X_train.columns if c.startswith(other + '_lag_') or c.startswith(other + '_diff_')]
            X_train.drop(columns=bad, errors='ignore', inplace=True)
            X_test .drop(columns=bad, errors='ignore', inplace=True)

    # Prepare multi-step shifted targets
    y_train_multi, y_train_raw = prepare_target_shifted(train_df[[variable]], variable, horizons)
    y_test_raw,  _            = prepare_target_shifted(test_df [[variable]], variable, horizons)

    # Align features with target horizon size and select features
    X_train = X_train.iloc[:len(y_train_multi)][selected_features].reset_index(drop=True)
    X_test  = X_test .iloc[:len(y_test_raw)][selected_features].reset_index(drop=True)

    # Train all models for the current station and variable
    for name, base in models.items():
        print(f"  ➤ Training {name}")

        # Apply scaling if required
        if name in scale_needed:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
            X_test  = pd.DataFrame(scaler.transform(X_test ), columns=selected_features)

        # Apply imputation for missing values
        imputer = SimpleImputer(strategy='mean')
        Xtr = pd.DataFrame(imputer.fit_transform(X_train), columns=selected_features)
        Xte = pd.DataFrame(imputer.transform(X_test ), columns=selected_features)

        # Train multi-output regressor
        reg = MultiOutputRegressor(base)
        reg.fit(Xtr, y_train_multi)

        # Save model
        outdir = os.path.join('models', variable)
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(reg, f"{outdir}/{name}_{variable}_{station_id}.pkl")

        # Make predictions on train and test
        Yp = reg.predict(Xte)
        Yt = reg.predict(Xtr)

        # Calculate R2 scores
        overall_r2 = r2_score(y_test_raw.values, Yp)
        train_r2   = r2_score(y_train_raw.values, Yt)

        # Evaluate each forecast horizon separately
        for i, h in enumerate(horizons):
            y_true = y_test_raw.iloc[:, i].values
            y_pred = Yp[:, i]
            datetimes = test_df.loc[y_test_raw.index, 'datetime'].reset_index(drop=True)
            future_times = datetimes + pd.to_timedelta(h, unit='h')

            # Calculate evaluation metrics and store results
            mae, mse, rmse, r2_step, evs, mape = evaluate(y_true, y_pred)
            all_metrics.append({
                'Variable': variable, 'Station': station_id, 'Model': name,
                'Step': h, 'Train R2': train_r2, 'Test R2': overall_r2,
                'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Step': r2_step,
                'EVS': evs, 'MAPE': mape
            })

            if station_id == 'merged':
                df_temp = pd.DataFrame({
                    'Datetime': future_times,
                    'Step': [h] * len(y_true),
                    'True': y_true,
                    'Predicted': y_pred
                })

                grouped = df_temp.groupby(['Datetime', 'Step']).agg({'True': 'mean', 'Predicted': 'mean'}).reset_index()


                for _, row in grouped.iterrows():
                    all_predictions.append({
                        'Variable': variable,
                        'Station': 'merged',
                        'Model': name,
                        'Step': row['Step'],
                        'Datetime': row['Datetime'],
                        'True': row['True'],
                        'Predicted': row['Predicted']
                    })

            else:
                for dt_i, t, p in zip(future_times, y_true, y_pred):
                    all_predictions.append({
                        'Variable': variable, 'Station': station_id, 'Model': name,
                        'Step': h, 'Datetime': dt_i, 'True': t, 'Predicted': p
                    })

def train_and_forecast(raw_csv_path, target_variables=None, per_station=True):
    """
    Loads raw SYNOP data, trains models (per station or merged), 
    performs forecasting, and saves results.

    Args:
        raw_csv_path (str): Path to the preprocessed CSV file containing SYNOP data.
        target_variables (list[str], optional): List of variables to forecast. 
                                                Defaults to ['temperature_c'].
        per_station (bool, optional): If True, trains separate models per station. 
                                      If False, merges data for global model.

    Saves:
        - Forecast results as CSV files in 'results/per_station' or 'results/merged'.
        - Performance metrics as 'metrics.csv'.
    """

    scale_needed = {'Linear Regression','Neural Network'}

    # Load and preprocess SYNOP data
    station_data = preprocess_synop_data(raw_csv_path,
                                         targets=target_variables,
                                         per_station=per_station)
    models = get_models()
    all_metrics = []

    outdir = f"results/{'per_station' if per_station else 'merged'}"
    os.makedirs(outdir, exist_ok=True)

    # Loop through each target variable
    for var in target_variables or ['temperature_c']:
        print(f"\n==================== {var.upper()} ====================\n")
        all_preds = []

        if per_station:
            train_list = []
            valid_features_by_station = {}

            # Loop over each station separately
            for sid, D in station_data.items():
                df = D['train_df'].copy(); df['station_id'] = sid
                train_list.append(df)

                # Identify shared features between train/test sets
                train_cols = set(D['train_df'].drop(columns=['datetime', var], errors='ignore').columns)
                test_cols  = set(D['test_df'].drop(columns=['datetime', var], errors='ignore').columns)
                shared_cols = train_cols & test_cols

                # Remove irrelevant features for current variable
                for other in ['temperature_c', 'humidity', 'wind_speed']:
                    if other != var:
                        shared_cols = {c for c in shared_cols if not c.startswith(other + '_lag_') and not c.startswith(other + '_diff_')}

                valid_features_by_station[sid] = shared_cols

            # Get common features across all stations
            common_features = set.intersection(*valid_features_by_station.values())
            merged_train = pd.concat(train_list, ignore_index=True)

            # Prepare merged training set
            X_all = merged_train[list(common_features)].copy()
            y_all = merged_train[[var]].dropna()
            X_all = X_all.loc[y_all.index]

            # Perform feature selection
            selected = mutual_info_feature_selection(
                X_all, y_all[var], top_k=20, min_mi=0.01, per_station=True, verbose=True
            )

            # Train and evaluate per station
            for sid, D in station_data.items():
                print(f"\n[Station {sid}]")
                run_training(D['train_df'], D['test_df'],
                             var, sid,
                             HORIZONS, models, scale_needed,
                             all_metrics, all_preds,
                             selected)
        else:
            # Global training (merged)
            D = station_data
            print("[Global]")
            tr, te = D['train_df'], D['test_df']

            # Drop unwanted columns from training data
            X = tr.drop(columns=['datetime', var], errors='ignore')
            for other in ['temperature_c', 'humidity', 'wind_speed']:
                if other != var:
                    bad = [c for c in X if c.startswith(other + '_lag_') or c.startswith(other + '_diff_')]
                    X.drop(columns=bad, errors='ignore', inplace=True)

            # Prepare and clean training features and targets
            y = tr[[var]].dropna()
            X = X.loc[y.index].select_dtypes(include=[np.number]).dropna()
            y = y.loc[X.index]

            # Perform feature selection
            selected = mutual_info_feature_selection(
                X, y[var], top_k=20, min_mi=0.01, per_station=per_station, verbose=True
            )

            # Include station-specific one-hot columns
            station_cols = [col for col in X.columns if col.startswith("station_") and col[8:].isdigit()]
            if station_cols:
                print("[INCLUDE]", station_cols)
                selected += station_cols

            # Train and evaluate for merged/global model
            run_training(tr, te, var, 'merged',
                         HORIZONS, models, scale_needed,
                         all_metrics, all_preds,
                         selected)
            
        # Save all predictions and metrics
        pd.DataFrame(all_preds).to_csv(f"{outdir}/predictions_{var}.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(f"{outdir}/metrics.csv", index=False)
    print("\nINFO: Training and forecasting completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models per station or merged.")
    parser.add_argument(
        "--per_station",
        action="store_true",
        help="Train separate models per station (default: merged)"
    )
    args = parser.parse_args()

    train_and_forecast(
        raw_csv_path="dataset/synop.csv",
        target_variables=["temperature_c"],
        per_station=args.per_station
    )
