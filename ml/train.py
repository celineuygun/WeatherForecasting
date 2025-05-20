import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
import joblib
from models import get_models
from preprocess import preprocess_synop_data
from utils import evaluate, prepare_target_shifted, mutual_info_feature_selection

def run_training(train_df, test_df, variable, station_id,
                 horizons, models, scale_needed,
                 all_metrics, all_predictions,
                 selected_features):

    # drop timestamp & target
    drop_cols = ['datetime', variable]
    train_df = train_df.dropna().reset_index(drop=True)
    test_df  = test_df.dropna().reset_index(drop=True)

    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test  = test_df .drop(columns=[c for c in drop_cols if c in test_df.columns])

    # remove other-variable lags/diffs
    for other in ['temperature_c','humidity','wind_speed']:
        if other != variable:
            bad = [c for c in X_train.columns if c.startswith(other + '_lag_') or c.startswith(other + '_diff_')]
            X_train.drop(columns=bad, errors='ignore', inplace=True)
            X_test .drop(columns=bad, errors='ignore', inplace=True)

    # build multioutput targets for only the specified horizons
    y_train_multi, y_train_raw = prepare_target_shifted(train_df[[variable]], variable, horizons)
    y_test_raw,  _            = prepare_target_shifted(test_df [[variable]], variable, horizons)

    # align features
    X_train = X_train.iloc[:len(y_train_multi)][selected_features].reset_index(drop=True)
    X_test  = X_test .iloc[:len(y_test_raw)][selected_features].reset_index(drop=True)

    for name, base in models.items():
        print(f"  ➤ Training {name}")
        if name in scale_needed:
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
            X_test  = pd.DataFrame(scaler.transform(X_test ), columns=selected_features)

        imputer = SimpleImputer(strategy='mean')
        Xtr = pd.DataFrame(imputer.fit_transform(X_train), columns=selected_features)
        Xte = pd.DataFrame(imputer.transform(X_test ), columns=selected_features)

        reg = MultiOutputRegressor(base)
        reg.fit(Xtr, y_train_multi)

        # save model
        outdir = os.path.join('models', variable)
        os.makedirs(outdir, exist_ok=True)
        joblib.dump(reg, f"{outdir}/{name}_{variable}_{station_id}.pkl")

        # predict
        Yp = reg.predict(Xte)
        Yt = reg.predict(Xtr)

        overall_r2 = r2_score(y_test_raw.values, Yp)
        train_r2   = r2_score(y_train_raw.values, Yt)

        # record metrics & predictions
        for i, h in enumerate(horizons):
            y_true = y_test_raw.iloc[:,i].values
            y_pred = Yp[:,i]
            idx    = y_test_raw.index
            dt     = test_df.loc[idx, 'datetime'] + pd.to_timedelta(h, unit='h')

            mae, mse, rmse, r2_step, evs, mape = evaluate(y_true, y_pred)
            all_metrics.append({
                'Variable':variable,'Station':station_id,'Model':name,
                'Step':h,'Train R2':train_r2,'Test R2':overall_r2,
                'MAE':mae,'MSE':mse,'RMSE':rmse,'R2 Step':r2_step,
                'EVS':evs,'MAPE':mape
            })
            for dt_i, t, p in zip(dt, y_true, y_pred):
                all_predictions.append({
                    'Variable':variable,'Station':station_id,'Model':name,
                    'Step':h,'Datetime':dt_i,'True':t,'Predicted':p
                })


def train_and_forecast(raw_csv_path, target_variables=None, per_station=True):
    horizons = [0, 3, 6]
    scale_needed = {'Linear Regression','Neural Network'}

    station_data = preprocess_synop_data(raw_csv_path,
                                        targets=target_variables,
                                        per_station=per_station)
    models      = get_models()
    all_metrics    = []

    outdir = f"results/{'per_station' if per_station else 'merged'}"
    os.makedirs(outdir, exist_ok=True)

    for var in target_variables or ['temperature_c']:
        print(f"\n==================== {var.upper()} ====================\n")
        all_preds = []

        if per_station:
            train_list = []
            for sid, D in station_data.items():
                df = D['train_df'].copy(); df['station_id']=sid
                train_list.append(df)
            merged_train = pd.concat(train_list,ignore_index=True)

            X_all = merged_train.drop(columns=['datetime',var],errors='ignore')
            for other in ['temperature_c','humidity','wind_speed']:
                if other!=var:
                    bad = [c for c in X_all if c.startswith(other+'_lag_') or c.startswith(other+'_diff_')]
                    X_all.drop(columns=bad,errors='ignore',inplace=True)
            y_all = merged_train[[var]].dropna()
            X_all = X_all.loc[y_all.index]

            selected = mutual_info_feature_selection(
                X_all, y_all[var], top_k=20, min_mi=0.01, verbose=True
            )

            for sid, D in station_data.items():
                print(f"\n[Station {sid}]")
                run_training(D['train_df'], D['test_df'],
                             var, sid,
                             horizons, models, scale_needed,
                             all_metrics, all_preds,
                             selected)
        else:
            D = station_data
            print("[Global]")
            tr, te = D['train_df'], D['test_df']

            X = tr.drop(columns=['datetime',var],errors='ignore')
            for other in ['temperature_c','humidity','wind_speed']:
                if other!=var:
                    bad = [c for c in X if c.startswith(other+'_lag_') or c.startswith(other+'_diff_')]
                    X.drop(columns=bad,errors='ignore',inplace=True)

            y = tr[[var]].dropna()
            X = X.loc[y.index].select_dtypes(include=[np.number]).dropna()
            y = y.loc[X.index]

            selected = mutual_info_feature_selection(
                X, y[var], top_k=20, min_mi=0.01, verbose=True
            )

            station_cols = [col for col in X.columns if col.startswith("station_") and col[8:].isdigit()]
            if station_cols:
                print("[INCLUDE]",station_cols)
                selected += station_cols

            run_training(tr, te, var, 'merged',
                         horizons, models, scale_needed,
                         all_metrics, all_preds,
                         selected)

        pd.DataFrame(all_preds).to_csv(f"{outdir}/predictions_{var}.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(f"{outdir}/metrics.csv", index=False)
    print("\nINFO: Training and forecasting completed.")

if __name__=='__main__':
    train_and_forecast('dataset/synop.csv',
                       target_variables=['temperature_c'],
                       per_station=True)