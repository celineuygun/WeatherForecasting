import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def evaluate_fusion(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def neural_network_fusion(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(0.01))
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=16, verbose=0, callbacks=[es])

    return model.predict(X_val, verbose=0).flatten(), y_val

def fusion_predictions(df, method="simple_avg", meta_model=None, verbose=False):
    df = df.copy()
    df['InstanceID'] = df.groupby(['Model']).cumcount()
    pivot = df.pivot(index='InstanceID', columns='Model', values='Predicted')
    y_true = df.drop_duplicates('InstanceID').set_index('InstanceID')['True'].loc[pivot.index]

    X = pivot.values
    y = y_true.values

    if method == "simple_avg":
        y_pred = pivot.mean(axis=1)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    elif method == "weighted_avg":
        r2_scores = {
            model: r2_score(
                df[df['Model'] == model].groupby('InstanceID')['True'].first(),
                df[df['Model'] == model].groupby('InstanceID')['Predicted'].first()
            ) for model in pivot.columns
        }
        weights = np.maximum(0, np.array(list(r2_scores.values())))
        weights /= weights.sum() + 1e-6
        y_pred = pivot.dot(weights)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    elif method == "error_weighted":
        errors = {
            model: mean_absolute_error(
                df[df['Model'] == model].groupby('InstanceID')['True'].first(),
                df[df['Model'] == model].groupby('InstanceID')['Predicted'].first()
            ) for model in pivot.columns
        }
        weights = 1 / (np.array(list(errors.values())) + 1e-6)
        weights /= weights.sum()
        y_pred = pivot.dot(weights)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    elif method in ["stacking", "meta_model"]:
        if meta_model is None:
            meta_model = (LinearRegression() if method == "stacking"
                          else RandomForestRegressor(n_estimators=100, random_state=42))
        meta_model = clone(meta_model)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        meta_model.fit(X_train, y_train)
        y_pred = meta_model.predict(X_test)
        return y_test, y_pred, evaluate_fusion(y_test, y_pred)

    elif method == "bayesian_avg":
        sampled_preds = [
            pivot.dot(np.random.dirichlet(np.ones(len(pivot.columns))))
            for _ in range(30)
        ]
        y_pred = pd.concat(sampled_preds, axis=1).mean(axis=1)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    elif method == "neural_fusion":
        y_pred, y_val = neural_network_fusion(X, y)
        return y_val, y_pred, evaluate_fusion(y_val, y_pred)

    else:
        raise ValueError(f"Unknown fusion method: {method}")

def run_fusion_analysis(variables=['temperature_c', 'humidity', 'wind_speed']):
    results_dir = "results/per_station"
    output_file = os.path.join(results_dir, "fusion_metrics.csv")
    os.makedirs(results_dir, exist_ok=True)

    fusion_methods = [
        'simple_avg', 'weighted_avg', 'error_weighted',
        'stacking', 'meta_model', 'bayesian_avg', 'neural_fusion'
    ]

    all_metrics = []

    for variable in variables:
        pred_file = os.path.join(results_dir, f"predictions_{variable}.csv")
        if not os.path.exists(pred_file):
            print(f"[ERROR] File not found: {pred_file}")
            continue

        df = pd.read_csv(pred_file)
        stations = df['Station'].unique()
        steps = sorted(df['Step'].unique())

        for station in stations:
            df_station = df[df['Station'] == station].copy()

            print(f"\n[Station {station}] Fusion Results for {variable}")
            for step in steps:
                df_step = df_station[df_station['Step'] == step].copy()
                print(f"\n  ➤ Step: t+{step}")
                print("  ----------------------------------------------------")
                print("  Method         |     MAE |    RMSE |     R2 |   MAPE")
                print("  ----------------------------------------------------")
                for method in fusion_methods:
                    try:
                        y_true, y_pred, metrics = fusion_predictions(
                            df_step,
                            method=method,
                            meta_model=LinearRegression() if method == 'stacking' else None
                        )
                        print(f"  {method:<14} | {metrics['MAE']:7.2f} | {metrics['RMSE']:7.2f} | {metrics['R2']:6.2f} | {metrics['MAPE']:6.2f}")
                        all_metrics.append({
                            'Variable': variable,
                            'Station': station,
                            'Step': step,
                            'Fusion Method': method,
                            **metrics
                        })
                    except Exception as e:
                        print(f"  {method:<14} | ERROR: {str(e)}")

    pd.DataFrame(all_metrics).to_csv(output_file, index=False)
    print(f"\nINFO: Fusion evaluation complete. Metrics saved to: {output_file}")

if __name__ == "__main__":
    run_fusion_analysis(variables=['temperature_c'])
