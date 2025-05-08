import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def add_input_lags(df, target_col, lags):
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        df[f"{target_col}_diff_{lag}"] = df[target_col] - df[target_col].shift(lag)
    return df

def add_temporal_features(df, datetime_col='datetime'):
    df['hour'] = df[datetime_col].dt.hour
    df['month'] = df[datetime_col].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_weather_regime(df, features=None, n_clusters=4):
    if features is None:
        features = ['temperature_c', 'sea_level_pressure', 'humidity', 'wind_speed']
    valid_rows = df[features].dropna()
    if len(valid_rows) < n_clusters:
        df['weather_regime'] = np.nan
        return df

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(valid_rows)
    clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(reduced)

    df['weather_regime'] = np.nan
    df.loc[valid_rows.index, 'weather_regime'] = clusters
    df['weather_regime'] = df['weather_regime'].astype('category')
    return df

def enrich_features(df):
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['temp_dew_diff'] = df['temperature_c'] - df['dew_point']
    es = 0.6108 * np.exp(17.27 * df['temperature_c'] / (df['temperature_c'] + 237.3))
    ea = 0.6108 * np.exp(17.27 * df['dew_point'] / (df['dew_point'] + 237.3))
    df['svp'] = es
    df['avp'] = ea
    df['vpd'] = es - ea

    if 'wind_dir_deg' in df.columns:
        radians = np.deg2rad(df['wind_dir_deg'])
        df['sin_wind_dir'] = np.sin(radians)
        df['cos_wind_dir'] = np.cos(radians)
        df['wind_dir_deg_lag_1'] = df['wind_dir_deg'].shift(1)
        df['wind_dir_deg_diff'] = df['wind_dir_deg'] - df['wind_dir_deg_lag_1']
        if df[['wind_speed', 'wind_dir_deg']].dropna().shape[0] >= 4:
            idx = df[['wind_speed', 'wind_dir_deg']].dropna().index
            df.loc[idx, 'wind_cluster'] = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(
                df.loc[idx, ['wind_speed', 'wind_dir_deg']]
            )

    df['humidity_x_temp'] = df['humidity'] * df['temperature_c']
    df['wind_x_vpd'] = df['wind_speed'] * df['vpd']
    if 'press_var_3h' in df.columns:
        df['wind_x_press3h'] = df['wind_speed'] * df['press_var_3h']

    df = add_weather_regime(df)
    return df

def handle_outliers(df, column, window=48, label=""):
    original_len = len(df)
    rolling_median = df[column].rolling(window=window, center=True).median()
    mad = (df[column] - rolling_median).abs().rolling(window=window).median()
    threshold = 4 * mad
    mask = (df[column] - rolling_median).abs() > threshold
    num_outliers = mask.sum()
    percent_outliers = 100 * num_outliers / original_len
    if num_outliers > 0:
        print(f"\n[OUTLIER HANDLING] {label} — {column}")
        print(f"  ➔ {num_outliers} outliers replaced ({percent_outliers:.2f}%)")
        print(df.loc[mask, [column]].head(3))
    df.loc[mask, column] = rolling_median[mask]
    return df

def validate_preprocessing(df, label=""):
    print(f"\n[VALIDATION] {label}")
    print(f"  ➔ Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  ➔ Shape: {df.shape}")
    print(f"  ➔ Columns: {df.columns.tolist()}")

def report_missing(df, name=""):
    total_missing = df.isna().sum()
    missing_cols = total_missing[total_missing > 0]
    if not missing_cols.empty:
        print(f"\n[MISSING DATA] {name}:")
        print(missing_cols)

def preprocess_one(df, targets, lags, label):
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.set_index('datetime')

    missing_ratio = df.isna().mean()
    to_drop = missing_ratio[missing_ratio > 0.7].index.tolist()
    if to_drop:
        print(f"\n[COLUMN DROPPING] {label} — Dropping columns with >70% missing: {to_drop}")
        df = df.drop(columns=to_drop)

    num_cols = [col for col in [
        'station_id','datetime','sea_level_pressure','press_var_3h','press_var_24h','baro_trend',
        'wind_dir','wind_speed','temperature_c','dew_point','humidity','visibility','cloud_cover',
        'station_pressure','latitude','longitude','altitude','rafale_10min','rafales_periode',
        'precip_1h','precip_3h','precip_6h','precip_12h','precip_24h'
    ] if col in df.columns]
    report_missing(df[num_cols], name=label)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].interpolate(method='time').ffill().bfill()
    df = df.reset_index()

    df = add_temporal_features(df)
    for t in targets:
        if t in df.columns:
            df = handle_outliers(df, t, window=48, label=label)
            df = add_input_lags(df, t, lags)
    df = enrich_features(df)
    df = df.dropna().reset_index(drop=True)
    validate_preprocessing(df, label)
    return df

def split_and_return(df, ratio=0.8):
    idx = int(len(df) * ratio)
    return {'train_df': df.iloc[:idx].copy(), 'test_df': df.iloc[idx:].copy()}

def preprocess_synop_data(path, lags=[1,3,6,12,24], targets=None, per_station=True):
    if targets is None:
        targets = ['temperature_c', 'humidity', 'wind_speed']

    df = pd.read_csv(path, sep=';', low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'], utc=True)

    cols = [
        'ID OMM station','datetime','Pression au niveau mer','Variation de pression en 3 heures',
        'Variation de pression en 24 heures','Type de tendance barométrique',
        'Direction du vent moyen 10 mn','Vitesse du vent moyen 10 mn','Température (°C)',
        'Point de rosée','Humidité','Visibilité horizontale','Nebulosité totale','Pression station',
        'Latitude','Longitude','Altitude','Rafale sur les 10 dernières minutes','Rafales sur une période',
        'Précipitations dans la dernière heure','Précipitations dans les 3 dernières heures',
        'Précipitations dans les 6 dernières heures','Précipitations dans les 12 dernières heures',
        'Précipitations dans les 24 dernières heures'
    ]
    df = df[cols]
    df.columns = [
        'station_id','datetime','sea_level_pressure','press_var_3h','press_var_24h','baro_trend',
        'wind_dir','wind_speed','temperature_c','dew_point','humidity','visibility','cloud_cover',
        'station_pressure','latitude','longitude','altitude','rafale_10min','rafales_periode',
        'precip_1h','precip_3h','precip_6h','precip_12h','precip_24h'
    ]

    df['dew_point'] = df['dew_point'] - 273.15
    df['baro_trend'] = df['baro_trend'].fillna(df['baro_trend'].mode().iloc[0]).astype(str)
    df['wind_dir'] = df['wind_dir'].ffill().bfill().astype(str)

    os.makedirs('preprocessed_dataset', exist_ok=True)
    dir_map = {'N':0,'NE':45,'E':90,'SE':135,'S':180,'SW':225,'W':270,'NW':315}

    data, all_dfs = {}, []
    for sid, grp in df.groupby('station_id'):
        g = grp.copy()
        g = pd.get_dummies(g, columns=['baro_trend'], prefix='trend')
        g['wind_dir_deg'] = pd.to_numeric(g['wind_dir'], errors='coerce')
        g['wind_dir_deg'] = g['wind_dir_deg'].fillna(g['wind_dir'].map(dir_map))

        proc = preprocess_one(g, targets, lags, f'station {sid}')
        splits = split_and_return(proc)
        train_df = splits['train_df']
        test_df  = splits['test_df']

        data[sid] = {'train_df': train_df, 'test_df': test_df}
        all_dfs.append(pd.concat([train_df, test_df]))

    full = pd.concat(all_dfs).sort_values("datetime").reset_index(drop=True)
    if not per_station:
        full = pd.get_dummies(full, columns=['station_id'], prefix='station')

    full.to_csv('preprocessed_dataset/preprocessed_synop.csv', index=False)
    if per_station:
        return data
    else:
        print("\n[MERGED DATA MODE] Using merged dataset for training.")
        return split_and_return(full)
