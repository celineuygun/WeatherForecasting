import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

LAGS_DEFAULT = [3, 6, 9, 24]
TRAIN_RATIO = 0.8
N_WEATHER_CLUSTERS = 4
MISSING_THRESHOLD = 0.7

def read_and_prepare(path):
    """
    Loads raw SYNOP CSV data, selects and renames columns, converts dew point,
    handles missing barometric trend values, and applies one-hot encoding.

    Args:
        path (str): File path to the raw CSV file.

    Returns:
        DataFrame: Cleaned and renamed raw dataset.
    """

    # Load CSV file and convert 'Date' column to datetime
    print("INFO: Loading raw data and renaming columns.")
    df = pd.read_csv(path, sep=';', low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'], utc=True)

    # Select only relevant columns
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

    # Rename columns to English equivalents
    df.columns = [
        'station_id', 'datetime', 'sea_level_pressure', 'press_var_3h', 'press_var_24h',
        'baro_trend', 'wind_dir', 'wind_speed', 'temperature_c', 'dew_point', 'humidity',
        'visibility', 'cloud_cover', 'station_pressure', 'latitude', 'longitude', 'altitude',
        'rafale_10min', 'precip_1h', 'precip_3h', 'precip_6h', 'precip_12h', 'precip_24h'
    ]
    # Convert dew point to Celsius
    df['dew_point'] = df['dew_point'] - 273.15

    # Fill barometric trend missing values without chained assignment
    baro_mode = df['baro_trend'].mode().iloc[0]
    df['baro_trend'] = df['baro_trend'].fillna(baro_mode)
    df = pd.get_dummies(df, columns=['baro_trend'], prefix='trend', dtype=int)

    print("  ✔ Loaded and renamed columns.")
    return df

def split_train_test(df):
    """
    Splits a DataFrame into training and testing sets based on datetime order.

    Args:
        df (DataFrame): Input DataFrame with datetime column.

    Returns:
        tuple: (train_df, test_df)
    """

    # Sort data by datetime and split based on TRAIN_RATIO
    print("INFO: Splitting into train and test.")
    df = df.sort_values('datetime')
    cutoff = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:cutoff].copy()
    test_df  = df.iloc[cutoff:].copy()
    print(f"  ✔ Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    return train_df, test_df

def drop_high_missing(df, label=""):
    """
    Drops columns from DataFrame that exceed the allowed missing data threshold.

    Args:
        df (DataFrame): Input DataFrame.
        label (str): Optional label for logging purposes.

    Returns:
        DataFrame: Reduced DataFrame with high-missing columns dropped.
    """

    # Identify columns with too many NaN values
    to_drop = df.isna().mean().loc[lambda x: x > MISSING_THRESHOLD].index.tolist()
    
    # Drop them if any
    if to_drop:
        print(f"  - [{label}] Dropping {len(to_drop)} cols >{MISSING_THRESHOLD*100:.0f}% missing: {to_drop}")
        return df.drop(columns=to_drop)
    return df

def fill_precip_zero(df, label=""):
    """
    Fills NaN values in precipitation-related columns with zero.

    Args:
        df (DataFrame): Input DataFrame.
        label (str): Optional label for logging.

    Returns:
        DataFrame: DataFrame with filled precipitation values.
    """

    # Identify precipitation-related columns
    prec_cols = [c for c in ['precip_1h','precip_3h','precip_6h','precip_12h','precip_24h','rafale_10min'] if c in df.columns]
    
    # Fill missing values with zero
    df[prec_cols] = df[prec_cols].fillna(0)
    print(f"  - [{label}] Filled precipitation NaNs in columns: {prec_cols}")
    return df

def interpolate_time(df, label=""):
    """
    Interpolates missing numeric values based on datetime index.

    Args:
        df (DataFrame): Input DataFrame with a datetime column.
        label (str): Optional label for logging.

    Returns:
        DataFrame: Interpolated DataFrame with filled numeric gaps.
    """

    # Set datetime as index for time-based interpolation
    df = df.set_index('datetime')

    # Find numeric columns with missing values
    numcols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].isna().any()]
    
    # Interpolate over time and apply forward/backward fill
    df[numcols] = df[numcols].interpolate(method='time').ffill().bfill()
    df = df.reset_index()
    print(f"  - [{label}] Interpolated columns with NaN: {numcols}")
    return df

def handle_outliers(df, targets, label=""):
    """
    Detects and replaces outliers using median absolute deviation over a rolling window.

    Args:
        df (DataFrame): Input DataFrame.
        targets (list): List of columns to process.
        label (str): Optional label for logging.

    Returns:
        DataFrame: DataFrame with outliers smoothed out.
    """

    for col in targets:
        # Compute rolling median and MAD
        med = df[col].rolling(48, center=True).median()
        mad = (df[col] - med).abs().rolling(48).median()

        # Detect outliers and replace with rolling median
        mask = (df[col] - med).abs() > 4 * mad
        if mask.any():
            cnt = mask.sum()
            pct = 100 * cnt / len(df)
            print(f"  - [{label}] Outliers in {col}: {cnt} pts ({pct:.2f}%) replaced")
            df.loc[mask, col] = med[mask]
    return df

def add_lags(df, targets, label=""):
    """
    Adds lagged and differenced versions of selected features.

    Args:
        df (DataFrame): Input DataFrame.
        targets (list): List of columns to lag.
        label (str): Optional label for logging.

    Returns:
        DataFrame: Extended DataFrame with lag and diff columns added.
    """

    for col in targets:
        for lag in LAGS_DEFAULT:
            # Add lagged feature and its difference from current
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
            df[f"{col}_diff_{lag}"] = df[col] - df[col].shift(lag)
    print(f"  - [{label}] Added lags and diffs for {targets}")
    return df

def add_temporal(df, label=""):
    """
    Adds cyclic temporal features (hour and month) using sine and cosine transforms.

    Args:
        df (DataFrame): Input DataFrame with datetime column.
        label (str): Optional label for logging.

    Returns:
        DataFrame: DataFrame with temporal features added.
    """

    # Encode hour of day and month of year as cyclic features
    df['sin_hour'] = np.sin(2*np.pi*df['datetime'].dt.hour/24)
    df['cos_hour'] = np.cos(2*np.pi*df['datetime'].dt.hour/24)
    df['sin_month'] = np.sin(2*np.pi*df['datetime'].dt.month/12)
    df['cos_month'] = np.cos(2*np.pi*df['datetime'].dt.month/12)
    print(f"  - [{label}] Added cyclic time features")
    return df

def compute_vpd(df, label=""):
    """
    Computes saturation vapor pressure, actual vapor pressure, and VPD (vapor pressure deficit).

    Args:
        df (DataFrame): Input DataFrame.
        label (str): Optional label for logging.

    Returns:
        DataFrame: DataFrame with 'svp', 'avp', and 'vpd' columns added.
    """

    # Compute saturation vapor pressure (es) and actual vapor pressure (ea)
    es = 0.6108 * np.exp(17.27*df['temperature_c']/(df['temperature_c']+237.3))
    ea = 0.6108 * np.exp(17.27*df['dew_point']/(df['dew_point']+237.3))

    # Derive VPD as the difference
    df['svp'], df['avp'], df['vpd'] = es, ea, es - ea
    print(f"  - [{label}] Computed VPD metrics")
    return df

def fit_weather_regime(train_df, feats, label=""):
    """
    Applies PCA and MiniBatchKMeans to cluster weather patterns in training data.

    Args:
        train_df (DataFrame): Training dataset.
        feats (list): List of feature columns to use.
        label (str): Optional label for logging.

    Returns:
        tuple: (pca, km) — Trained PCA and KMeans models.
    """

    # Drop NaNs in clustering features
    valid = train_df[feats].dropna()

    # Fit PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42).fit(valid)
    proj = pca.transform(valid)

    # Fit KMeans to identify weather regimes
    km = MiniBatchKMeans(n_clusters=N_WEATHER_CLUSTERS, random_state=42).fit(proj)

    # Assign regime labels
    train_df.loc[valid.index, 'weather_regime'] = km.predict(proj)
    print(f"  - [{label}] Fitted weather regime clustering")
    return pca, km

def apply_weather_regime(df, pca, km, feats, label=""):
    """
    Projects features using PCA and assigns weather regime clusters using trained KMeans.

    Args:
        df (DataFrame): Input DataFrame to label.
        pca (PCA): Trained PCA model.
        km (KMeans): Trained KMeans model.
        feats (list): Features to project.
        label (str): Optional label for logging.

    Returns:
        DataFrame: DataFrame with new 'weather_regime' column.
    """

    # Apply PCA transform and KMeans clustering to assign weather regimes
    valid = df[feats].dropna()
    proj = pca.transform(valid)
    df.loc[valid.index, 'weather_regime'] = km.predict(proj)
    print(f"  - [{label}] Applied weather regime clustering")
    return df

def preprocess_synop_data(path, targets=None, per_station=True):
    """
    Preprocesses raw SYNOP meteorological data for training/testing.

    Steps include:
    - Missing value handling
    - Interpolation
    - Lag and temporal feature generation
    - Outlier smoothing
    - Vapor pressure calculations
    - PCA-based weather regime clustering

    Args:
        path (str): Path to raw SYNOP CSV file.
        targets (list[str], optional): List of target variables to forecast.
        per_station (bool): Whether to process data per station or globally.

    Returns:
        dict or DataFrame:
            - If per_station=True: dict of station_id -> {'train_df', 'test_df'}
            - If per_station=False: {'train_df', 'test_df'} with global dataset
    """

    if targets is None:
        targets = ['temperature_c','humidity','wind_speed']

    # Read and clean raw SYNOP CSV data
    df_raw = read_and_prepare(path)

    # Define features used for clustering (targets + sea level pressure)
    feats = targets + ['sea_level_pressure']

    if per_station:
        print("\nINFO: Processing each station separately.")
        output, merged = {}, []

        # Loop over each station group
        for sid, grp in df_raw.groupby('station_id'):
            print(f"\n[Station {sid}]")

            # Split station data into train and test
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

            # Store preprocessed data
            output[sid] = {'train_df': tr, 'test_df': te}

            # Append to merged list
            merged.append(pd.concat([tr, te]))

        print("\nINFO: Saving merged dataset.")
        full = pd.concat(merged).sort_values('datetime').reset_index(drop=True)
        full.to_csv('dataset/preprocessed_synop.csv', index=False)
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
    full.to_csv('dataset/preprocessed_synop.csv', index=False)
    print(f"  ✔ Saved merged dataset ({len(full)} rows)")
    print("\nINFO: Preprocessing complete.")
    return {'train_df': tr, 'test_df': te}
