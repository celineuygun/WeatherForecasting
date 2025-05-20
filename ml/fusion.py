import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from utils import mean_absolute_percentage_error, evaluate

def model_fusion(df, method, step):
    df_step = df[df['Step'] == step].copy()
    df_step['InstanceID'] = df_step.groupby(['Model', 'Step']).cumcount()

    pivot = df_step.pivot_table(index='InstanceID', columns='Model', values='Predicted', aggfunc='mean')
    true_vals = df_step[['InstanceID', 'True']].drop_duplicates(subset='InstanceID').set_index('InstanceID')['True']
    pivot = pivot.dropna()
    y_true = true_vals.loc[pivot.index]

    if method == 'fedavg':
        y_pred = pivot.mean(axis=1)

    elif method == 'fedprox':
        stds = pivot.std()
        weights = 1 / (stds + 1e-6)
        weights /= weights.sum()
        y_pred = pivot.dot(weights)

    elif method == 'fedatt':
        r2_scores = {
            model: r2_score(
                df_step[df_step['Model'] == model].groupby('InstanceID')['True'].first(),
                df_step[df_step['Model'] == model].groupby('InstanceID')['Predicted'].first()
            ) for model in pivot.columns
        }
        attn_weights = np.exp(list(r2_scores.values()))
        attn_weights /= attn_weights.sum()
        y_pred = pivot.dot(attn_weights)

    elif method == 'fedbe':
        sampled_preds = [
            pivot.dot(np.random.dirichlet(np.ones(len(pivot.columns))))
            for _ in range(30)
        ]
        y_pred = pd.concat(sampled_preds, axis=1).mean(axis=1)

    elif method == 'feddyn':
        variances = pivot.var()
        weights = 1 / (variances + 1e-6)
        weights /= weights.sum()
        y_pred = pivot.dot(weights)

    elif method == 'fedfair':
        mape_scores = {
            model: mean_absolute_percentage_error(
                df_step[df_step['Model'] == model].groupby('InstanceID')['True'].first(),
                df_step[df_step['Model'] == model].groupby('InstanceID')['Predicted'].first()
            ) for model in pivot.columns
        }
        fairness_weights = 1 / (np.array(list(mape_scores.values())) + 1e-6)
        fairness_weights /= fairness_weights.sum()
        y_pred = pivot.dot(fairness_weights)

    elif method == 'fedmeta':
        meta_model = LinearRegression().fit(pivot.values, y_true.values)
        y_pred = pd.Series(meta_model.predict(pivot.values), index=pivot.index)

    elif method == 'fedcluster':
        km = KMeans(n_clusters=2, random_state=42)
        cluster_labels = km.fit_predict(pivot.T)
        cluster_preds = []
        for label in np.unique(cluster_labels):
            models_in_cluster = pivot.columns[cluster_labels == label]
            cluster_preds.append(pivot[models_in_cluster].mean(axis=1))
        y_pred = sum(cluster_preds) / len(cluster_preds)

    else:
        raise ValueError("Unknown fusion method")

    return evaluate(y_true.values, y_pred.values)

def run_fusion_analysis(variables=['temperature_c', 'humidity', 'wind_speed']):
    results_dir = "results/per_station"
    os.makedirs(results_dir, exist_ok=True)

    fusion_methods = [
        'fedavg', 'fedprox', 'fedatt', 'fedbe', 'feddyn',
        'fedfair', 'fedmeta', 'fedcluster'
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

            print(f"\n=== Fusion Results for: {variable} | Station: {station} ===")
            for step in steps:
                print(f"\n  ➤ Step: t+{step}")
                print("  -------------------------------------------")
                print("  Method       |     MAE |    RMSE |     R2 |   MAPE")
                print("  -------------------------------------------")
                for method in fusion_methods:
                    try:
                        mae, mse, rmse, r2, evs, mape = model_fusion(df_station, method, step)
                        print(f"  {method:<12} | {mae:7.2f} | {rmse:7.2f} | {r2:6.2f} | {mape:6.2f}")
                        all_metrics.append({
                            'Variable': variable,
                            'Station': station,
                            'Step': step,
                            'Fusion Method': method,
                            'MAE': mae,
                            'MSE': mse,
                            'RMSE': rmse,
                            'R2': r2,
                            'EVS': evs,
                            'MAPE': mape
                        })
                    except Exception as e:
                        print(f"  {method:<12} | ERROR: {str(e)}")

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(results_dir, "fusion_metrics.csv"), index=False)
    print("\nFusion evaluation complete. Metrics saved to fusion_metrics.csv.")

if __name__ == "__main__":
    run_fusion_analysis(variables=['temperature_c'])
