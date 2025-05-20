import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

LAGS_DEFAULT = [1, 2, 3, 4, 5, 6, 24]
TRAIN_RATIO = 0.8
N_WEATHER_CLUSTERS = 4
MISSING_THRESHOLD = 0.7

def read_and_prepare(path):
    print("INFO: Loading raw data and renaming columns.")
    df = pd.read_csv(path, sep=';', low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'], utc=True)
    cols = [
        'ID OMM station', 'datetime', 'Pression au niveau mer',
        'Variation de pression en 3 heures', 'Variation de pression en 24 heures',
        'Type de tendance barométrique', 'Direction du vent moyen 10 mn',
        'Vitesse du vent moyen 10 mn', 'Température (°C)', 'Point de rosée', 'Humidité',
        'Visibilité horizontale', 'Nebulosité totale', 'Pression station', 'Latitude', 'Longitude',
        'Altitude', 'Rafale sur les 10 dernières minutes',
        'Précipitations dans la dernière heure',
        'Précipitations dans les 3 dernières heures',
        'Précipitations dans les 6 dernières heures',
        'Précipitations dans les 12 dernières heures',
        'Précipitations dans les 24 dernières heures'
    ]
    df = df[cols]

    df.columns = [
        'station_id', 'datetime', 'sea_level_pressure', 'press_var_3h', 'press_var_24h',
        'baro_trend', 'wind_dir', 'wind_speed', 'temperature_c', 'dew_point', 'humidity',
        'visibility', 'cloud_cover', 'station_pressure', 'latitude', 'longitude', 'altitude',
        'rafale_10min', 'precip_1h', 'precip_3h', 'precip_6h', 'precip_12h', 'precip_24h'
    ]
    # Convert dew point
    df['dew_point'] = df['dew_point'] - 273.15
    # Fill barometric trend missing values without chained assignment
    baro_mode = df['baro_trend'].mode().iloc[0]
    df['baro_trend'] = df['baro_trend'].fillna(baro_mode)
    df = pd.get_dummies(df, columns=['baro_trend'], prefix='trend', dtype=int)

    print("  ✔ Loaded and renamed columns.")
    return df

def split_train_test(df):
    print("INFO: Splitting into train and test.")
    df = df.sort_values('datetime')
    cutoff = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:cutoff].copy()
    test_df  = df.iloc[cutoff:].copy()
    print(f"  ✔ Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    return train_df, test_df

def drop_high_missing(df, label=""):
    to_drop = df.isna().mean().loc[lambda x: x > MISSING_THRESHOLD].index.tolist()
    if to_drop:
        print(f"  - [{label}] Dropping {len(to_drop)} cols >{MISSING_THRESHOLD*100:.0f}% missing: {to_drop}")
        return df.drop(columns=to_drop)
    return df

def fill_precip_zero(df, label=""):
    prec_cols = [c for c in ['precip_1h','precip_3h','precip_6h','precip_12h','precip_24h','rafale_10min'] if c in df.columns]
    df[prec_cols] = df[prec_cols].fillna(0)
    print(f"  - [{label}] Filled precipitation NaNs in columns: {prec_cols}")
    return df

def interpolate_time(df, label=""):
    df = df.set_index('datetime')
    numcols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].isna().any()]
    df[numcols] = df[numcols].interpolate(method='time').ffill().bfill()
    df = df.reset_index()
    print(f"  - [{label}] Interpolated columns with NaN: {numcols}")
    return df

def handle_outliers(df, targets, label=""):
    for col in targets:
        med = df[col].rolling(48, center=True).median()
        mad = (df[col] - med).abs().rolling(48).median()
        mask = (df[col] - med).abs() > 4 * mad
        if mask.any():
            cnt = mask.sum()
            pct = 100 * cnt / len(df)
            print(f"  - [{label}] Outliers in {col}: {cnt} pts ({pct:.2f}%) replaced")
            df.loc[mask, col] = med[mask]
    return df

def add_lags(df, targets, label=""):
    for col in targets:
        for lag in LAGS_DEFAULT:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            df[f"{col}_diff_{lag}"] = df[col] - df[col].shift(lag)
    print(f"  - [{label}] Added lags and diffs for {targets}")
    return df

def add_temporal(df, label=""):
    df['sin_hour'] = np.sin(2*np.pi*df['datetime'].dt.hour/24)
    df['cos_hour'] = np.cos(2*np.pi*df['datetime'].dt.hour/24)
    df['sin_month'] = np.sin(2*np.pi*df['datetime'].dt.month/12)
    df['cos_month'] = np.cos(2*np.pi*df['datetime'].dt.month/12)
    print(f"  - [{label}] Added cyclic time features")
    return df

def compute_vpd(df, label=""):
    es = 0.6108 * np.exp(17.27*df['temperature_c']/(df['temperature_c']+237.3))
    ea = 0.6108 * np.exp(17.27*df['dew_point']/(df['dew_point']+237.3))
    df['svp'], df['avp'], df['vpd'] = es, ea, es - ea
    print(f"  - [{label}] Computed VPD metrics")
    return df

def fit_weather_regime(train_df, feats, label=""):
    valid = train_df[feats].dropna()
    pca = PCA(n_components=2, random_state=42).fit(valid)
    proj = pca.transform(valid)
    km = MiniBatchKMeans(n_clusters=N_WEATHER_CLUSTERS, random_state=42).fit(proj)
    train_df.loc[valid.index, 'weather_regime'] = km.predict(proj)
    print(f"  - [{label}] Fitted weather regime clustering")
    return pca, km

def apply_weather_regime(df, pca, km, feats, label=""):
    valid = df[feats].dropna()
    proj = pca.transform(valid)
    df.loc[valid.index, 'weather_regime'] = km.predict(proj)
    print(f"  - [{label}] Applied weather regime clustering")
    return df

def preprocess_synop_data(path, targets=None, per_station=True):
    """
    Preprocess SYNOP data, either per station or merged global dataset.

    Returns:
      - If per_station=True: dict of station_id -> {'train_df', 'test_df'}
      - If per_station=False: {'train_df', 'test_df'} for merged data
    """
    if targets is None:
        targets = ['temperature_c','humidity','wind_speed']

    df_raw = read_and_prepare(path)
    os.makedirs('preprocessed_dataset', exist_ok=True)
    feats = targets + ['sea_level_pressure']

    if per_station:
        print("\nINFO: Processing each station separately.")
        output, merged = {}, []
        for sid, grp in df_raw.groupby('station_id'):
            print(f"\n[Station {sid}]")
            tr_raw, te_raw = split_train_test(grp.copy())

            # Train
            print("\n  a) Train preprocessing:")
            tr = drop_high_missing(tr_raw, 'train')
            tr = fill_precip_zero(tr, 'train')
            tr = interpolate_time(tr, 'train')
            tr = handle_outliers(tr, targets, 'train')
            tr = add_lags(tr, targets, 'train')
            tr = add_temporal(tr, 'train')
            tr = compute_vpd(tr, 'train')
            pca, km = fit_weather_regime(tr, feats, 'train')
            tr = apply_weather_regime(tr, pca, km, feats, 'train')
            tr.dropna(inplace=True)
            print(f"  ✔ Train subset ready ({len(tr)} rows)")

            # Test
            print("\n  b) Test preprocessing:")
            te = drop_high_missing(te_raw, 'test')
            te = fill_precip_zero(te, 'test')
            te = interpolate_time(te, 'test')
            te = handle_outliers(te, targets, 'test')
            te = add_lags(te, targets, 'test')
            te = add_temporal(te, 'test')
            te = compute_vpd(te, 'test')
            te = apply_weather_regime(te, pca, km, feats, 'test')
            te.dropna(inplace=True)
            print(f"  ✔ Test subset ready ({len(te)} rows)")

            output[sid] = {'train_df': tr, 'test_df': te}
            merged.append(pd.concat([tr, te]))

        print("\nINFO: Saving merged dataset.")
        full = pd.concat(merged).sort_values('datetime').reset_index(drop=True)
        full.to_csv('preprocessed_dataset/preprocessed_synop.csv', index=False)
        print(f"  ✔ Saved merged dataset ({len(full)} rows)")
        print("\nINFO: Preprocessing complete.")
        return output

    print("\nINFO: Processing merged dataset.")
    tr_raw, te_raw = split_train_test(df_raw.copy())

    # Train
    print("\n  a) Train preprocessing:")
    tr = drop_high_missing(tr_raw, 'train')
    tr = fill_precip_zero(tr, 'train')
    tr = interpolate_time(tr, 'train')
    tr = handle_outliers(tr, targets, 'train')
    tr = add_lags(tr, targets, 'train')
    tr = add_temporal(tr, 'train')
    tr = compute_vpd(tr, 'train')
    pca, km = fit_weather_regime(tr, feats, 'train')
    tr = apply_weather_regime(tr, pca, km, feats, 'train')
    tr.dropna(inplace=True)
    tr = pd.get_dummies(tr, columns=['station_id'], prefix='station', dtype=int)
    print(f"  ✔ Global train ready ({len(tr)} rows)")

    # Test
    print("\n  b) Test preprocessing:")
    te = drop_high_missing(te_raw, 'test')
    te = fill_precip_zero(te, 'test')
    te = interpolate_time(te, 'test')
    te = handle_outliers(te, targets, 'test')
    te = add_lags(te, targets, 'test')
    te = add_temporal(te, 'test')
    te = compute_vpd(te, 'test')
    te = apply_weather_regime(te, pca, km, feats, 'test')
    te.dropna(inplace=True)
    te = pd.get_dummies(te, columns=['station_id'], prefix='station', dtype=int)
    print(f"  ✔ Global test ready ({len(te)} rows)")

    print("\nINFO: Saving merged dataset.")
    full = pd.concat([tr, te]).sort_values('datetime').reset_index(drop=True)
    full.to_csv('preprocessed_dataset/preprocessed_synop.csv', index=False)
    print(f"  ✔ Saved merged dataset ({len(full)} rows)")
    print("\nINFO: Preprocessing complete.")
    return {'train_df': tr, 'test_df': te}
