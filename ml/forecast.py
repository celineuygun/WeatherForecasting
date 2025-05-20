import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import os

csv_path = "results/per_station/predictions_temperature_c.csv"
target_datetime = pd.to_datetime("2025-04-08 18:00:00+00:00")

stations = {
    "7761": {"name": "Ajaccio", "lat": 41.918, "lon": 8.792667, "alt": 5},
    "7790": {"name": "Bastia",  "lat": 42.6975, "lon": 9.4500,   "alt": 10}
}

df = pd.read_csv(csv_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Station'] = df['Station'].astype(str)

df = df[df['Datetime'] == target_datetime]
if df.empty:
    print(f"No data found for {target_datetime}")
    exit()

steps = sorted(df['Step'].unique())
output_dir = "plots/temperature_c"
os.makedirs(output_dir, exist_ok=True)

for step in steps:
    df_step = df[df['Step'] == step]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=crs.PlateCarree())
    ax.set_extent([7.8, 10.2, 41.1, 43.1], crs=crs.PlateCarree())
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.set_title(f"T+{step} Forecasts - {target_datetime.strftime('%Y-%m-%d %H:%M UTC')}")

    for sid, info in stations.items():
        lat, lon = info["lat"], info["lon"]
        station_name = info["name"]

        ax.plot(lon, lat, marker='o', color='black', markersize=6, transform=crs.PlateCarree())

        preds = df_step[df_step['Station'] == sid]
        if preds.empty:
            continue

        true_val = preds.iloc[0]['True']
        label_lines = [f"{station_name}", f"{alt} m", f"True: {true_val:.1f}°C"]

        for _, row in preds.iterrows():
            model = row['Model']
            pred = row['Predicted']
            label_lines.append(f"{model}: {pred:.1f}°C")


        label_text = "\n".join(label_lines)
        ax.text(lon + 0.05, lat + 0.05, label_text,
                transform=crs.PlateCarree(),
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.savefig(f"{output_dir}/forecast_step_{step}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"INFO: Plots saved to {output_dir}")
