import pandas as pd
import numpy as np
import os
from collections import defaultdict

def add_input_lags(df, target_col='temperature_c', lags=[1, 3, 6, 12, 24]):
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

def add_temporal_features(df, datetime_col='datetime'):
    df['hour'] = df[datetime_col].dt.hour
    df['month'] = df[datetime_col].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def validate_preprocessing(df, station_id):
    print(f"\nINFO: Validating station {station_id}")

    print(f"  ➤ Original length: {len(df)}")
    print(f"  ➤ Length after preprocessing: {len(df)}")
    if 'temperature_c_lag_1' in df.columns:
        i = 20
        if len(df) > i:
            actual = df['temperature_c'].iloc[i - 1]
            lagged = df['temperature_c_lag_1'].iloc[i]
            print(f"  ➤ Lag check (i={i}): t-1={actual}, lag_1={lagged}, match={np.isclose(actual, lagged)}")
        else:
            print(f"ERROR: Not enough rows ({len(df)}) to validate lag at index {i}")


    print(f"  ➤ Datetime range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  ➤ Columns: {df.columns.tolist()}")

def preprocess_synop_data(filepath, target_col='temperature_c', lags=[1,3,6,12,24]):
    df = pd.read_csv(filepath, delimiter=';', low_memory=False)
    df['datetime'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

    df = df[['WMO Station ID', 'datetime',
             'Sea level pressure', 'Humidity', '10-min mean wind speed',
             'Dew point', 'Total cloud cover', 'Horizontal visibility',
             'Temperature (C)']]
    
    df.columns = ['station_id', 'datetime',
                  'sea_level_pressure', 'humidity', 'wind_speed',
                  'dew_point', 'cloud_cover', 'visibility', 'temperature_c']

    df = df.sort_values(by=['station_id', 'datetime'])
    df = df.dropna()

    station_data = {}
    all_records = []

    for station_id, group in df.groupby('station_id'):
        group = group.reset_index(drop=True)
        group = add_input_lags(group, target_col=target_col, lags=lags)
        group = add_temporal_features(group, datetime_col='datetime')
        group = group.dropna()

        original_len = len(group)
        validate_preprocessing(group, station_id)

        split_idx = int(len(group) * 0.8)
        train_df = group.iloc[:split_idx].copy()
        test_df = group.iloc[split_idx:].copy()

        train_df['split'] = 'train'
        test_df['split'] = 'test'

        all_records.append(train_df)
        all_records.append(test_df)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        assert X_train.index.equals(y_train.index), "X_train ve y_train indeks uyumsuz"
        assert X_test.index.equals(y_test.index), "X_test ve y_test indeks uyumsuz"

        station_data[station_id] = {
            'X_train': X_train.drop(columns=['datetime']),
            'X_test': X_test.drop(columns=['datetime']),
            'y_train': y_train,
            'y_test': y_test,
            'train_datetime': train_df['datetime'],
            'test_datetime': test_df['datetime']
        }
        print(f"  ➤ Station {station_id}: Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessed_df = pd.concat(all_records, ignore_index=True)
    os.makedirs("preprocessed_dataset", exist_ok=True)
    preprocessed_df.to_csv("preprocessed_dataset/preprocessed_synop.csv", index=False)
    print("INFO: All preprocessed data saved to: preprocessed_dataset/preprocessed_synop.csv")

    return station_data