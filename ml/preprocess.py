import pandas as pd
import numpy as np
import os

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
    print(f"  ➤ Datetime range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  ➤ Number of rows: {len(df)}")
    print(f"  ➤ Number of columns: {len(df.columns)}")
    print(f"  ➤ Columns: {df.columns.tolist()}")

def preprocess_synop_data(filepath, lags=[1,3,6,12,24], target_variables=None):
    if target_variables is None:
        target_variables = ['temperature_c', 'humidity', 'wind_speed']

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
    all_processed_rows = []

    for station_id, group in df.groupby('station_id'):
        group = group.reset_index(drop=True)
        group = add_temporal_features(group, datetime_col='datetime')

        for target in target_variables:
            group = add_input_lags(group, target_col=target, lags=lags)

        group = group.dropna()
        validate_preprocessing(group, station_id)

        split_idx = int(len(group) * 0.8)
        train_df = group.iloc[:split_idx].copy()
        test_df = group.iloc[split_idx:].copy()

        station_data[station_id] = {
            'train_df': train_df,
            'test_df': test_df,
        }

        all_processed_rows.append(group)


    final_df = pd.concat(all_processed_rows, axis=0)
    merged_out = "preprocessed_dataset"
    os.makedirs(merged_out, exist_ok=True)
    final_df.to_csv(f"{merged_out}/preprocessed_synop.csv", index=False)
    print(f"\nINFO: All preprocessed data saved to: {merged_out}/preprocessed_synop.csv")

    return station_data
