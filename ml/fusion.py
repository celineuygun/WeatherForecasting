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
    """
    Calculates common regression metrics for model evaluation.

    Metrics computed:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error)
    - R² (Coefficient of Determination)
    - MAPE (Mean Absolute Percentage Error)

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values by the fusion model.

    Returns:
        dict: Dictionary containing MAE, RMSE, R2, and MAPE.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def neural_network_fusion(X, y):
    """
    Trains a shallow neural network (MLP) to learn model fusion weights.

    The model is trained on a subset of the prediction data and validated on the rest.
    It consists of 2 hidden layers with ReLU activation and uses MSE loss.

    Args:
        X (ndarray): Input features (model predictions).
        y (ndarray): True target values.

    Returns:
        tuple:
            - y_pred (np.ndarray): Predicted values on validation set.
            - y_val  (np.ndarray): Ground truth values on validation set.
    """

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build neural network
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile and train with early stopping
    model.compile(loss='mse', optimizer=Adam(0.01))
    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss')

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=16, verbose=0, callbacks=[es])

    return model.predict(X_val, verbose=0).flatten(), y_val

def fusion_predictions(df, method="simple_avg", meta_model=None, verbose=False):
    """
    Applies one of several fusion strategies to combine model predictions.

    Supported fusion methods:
    - "simple_avg"     : Uniform averaging
    - "weighted_avg"   : Weighted by R² score
    - "error_weighted" : Inverse-weighted by MAE
    - "stacking"       : Linear regression on model predictions
    - "meta_model"     : Custom meta-learner (e.g., RandomForest)
    - "bayesian_avg"   : Dirichlet-based probabilistic averaging
    - "neural_fusion"  : Neural network fusion

    Args:
        df (DataFrame): DataFrame containing predictions per model with 'True' values.
        method (str): Fusion strategy to use.
        meta_model (sklearn regressor): Optional meta-model to use (if applicable).
        verbose (bool): If True, prints internal steps.

    Returns:
        tuple:
            - y_true (np.ndarray): Ground truth values.
            - y_pred (np.ndarray): Fused predicted values.
            - metrics (dict): Evaluation metrics for the fusion.
    """

    df = df.copy()

    # Create a unique ID for each instance
    df['InstanceID'] = df.groupby(['Model']).cumcount()
    
    # Pivot to wide format: rows = instances, columns = models
    pivot = df.pivot(index='InstanceID', columns='Model', values='Predicted')
    y_true = df.drop_duplicates('InstanceID').set_index('InstanceID')['True'].loc[pivot.index]

    X = pivot.values
    y = y_true.values

    # Simple average fusion
    if method == "simple_avg":
        y_pred = pivot.mean(axis=1)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    # Weighted average based on R² scores
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

    # Weighted average based on inverse MAE
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

    # Stacking or meta-model approach
    elif method in ["stacking", "meta_model"]:
        # Default to linear or random forest
        if meta_model is None:
            meta_model = (LinearRegression() if method == "stacking"
                          else RandomForestRegressor(n_estimators=100, random_state=42))
        meta_model = clone(meta_model)

        # Train meta-model on training split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        meta_model.fit(X_train, y_train)
        y_pred = meta_model.predict(X_test)
        return y_test, y_pred, evaluate_fusion(y_test, y_pred)

    # Bayesian average using Dirichlet sampling
    elif method == "bayesian_avg":
        sampled_preds = [
            pivot.dot(np.random.dirichlet(np.ones(len(pivot.columns))))
            for _ in range(30)
        ]
        y_pred = pd.concat(sampled_preds, axis=1).mean(axis=1)
        return y, y_pred.values, evaluate_fusion(y, y_pred.values)

    # Neural network fusion
    elif method == "neural_fusion":
        y_pred, y_val = neural_network_fusion(X, y)
        return y_val, y_pred, evaluate_fusion(y_val, y_pred)

    else:
        raise ValueError(f"Unknown fusion method: {method}")

def run_fusion_analysis(variables=['temperature_c', 'humidity', 'wind_speed']):
    """
    Runs fusion-based evaluation for each station, step, and variable.

    For each variable and station:
    - Loads corresponding prediction file.
    - Applies all supported fusion methods.
    - Evaluates and prints performance metrics.
    - Saves results to CSV in 'results/per_station/fusion_metrics.csv'.

    Args:
        variables (list[str]): List of target variables to evaluate fusion on.
    """
    
    results_dir = "results/per_station"
    output_file = os.path.join(results_dir, "fusion_metrics.csv")
    os.makedirs(results_dir, exist_ok=True)

    fusion_methods = [
        'simple_avg', 'weighted_avg', 'error_weighted',
        'stacking', 'meta_model', 'bayesian_avg', 'neural_fusion'
    ]

    all_metrics = []

    # Iterate over variables and forecast steps
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
                        # Run fusion and evaluate
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
