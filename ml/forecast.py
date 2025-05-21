import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import os

# Config
per_station = False
mode = "per_station" if per_station else "merged"
csv_path = f"results/{mode}/predictions_temperature_c.csv"
target_datetime = pd.to_datetime("2025-04-08 12:00:00+00:00")

# Load forecast data
df = pd.read_csv(csv_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Station'] = df['Station'].astype(str)
df = df[df['Datetime'] == target_datetime]

if df.empty:
    print(f"No data found for {target_datetime}")
    exit()

steps = sorted(df['Step'].unique())
output_dir = f"plots/temperature_c/{mode}"
os.makedirs(output_dir, exist_ok=True)

# Station locations
stations_meta = {
    "7761": {"name": "Ajaccio", "lat": 41.918, "lon": 8.792667, "alt": 5},
    "7790": {"name": "Bastia",  "lat": 42.540667, "lon": 9.485167, "alt": 10},
}

# Midpoint for merged display
lat1, lon1 = stations_meta["7761"]["lat"], stations_meta["7761"]["lon"]
lat2, lon2 = stations_meta["7790"]["lat"], stations_meta["7790"]["lon"]
mid_lat = (lat1 + lat2) / 2
mid_lon = (lon1 + lon2) / 2
mid_alt = (stations_meta["7761"]["alt"] + stations_meta["7790"]["alt"]) / 2

# Plot loop
for step in steps:
    df_step = df[df['Step'] == step]

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=crs.PlateCarree())
    ax.set_extent([7.8, 10.2, 41.1, 43.1], crs=crs.PlateCarree())
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    ax.set_title(
        f"T+{step} Forecasts - {target_datetime.strftime('%Y-%m-%d %H:%M UTC')} ({mode.replace('_', ' ').title()})"
    )

    # Plot station markers
    for sid, info in stations_meta.items():
        ax.plot(info['lon'], info['lat'], marker='o', color='black', markersize=6, transform=crs.PlateCarree())
        ax.text(info['lon'] - 0.18, info['lat'] - 0.06, info['name'],
                transform=crs.PlateCarree(), fontsize=7, color='gray')

    if per_station:
        for sid in df_step['Station'].unique():
            if sid not in stations_meta:
                continue
            info = stations_meta[sid]
            lat, lon, alt = info["lat"], info["lon"], info["alt"]

            preds = df_step[df_step['Station'] == sid]
            if preds.empty:
                continue

            true_val = preds.iloc[0]['True']
            label_lines = [f"{alt} m", f"True: {true_val:.1f}°C"]

            for model in preds['Model'].unique():
                pred = preds[preds['Model'] == model]['Predicted'].iloc[0]
                label_lines.append(f"{model}: {pred:.1f}°C")

            label_text = "\n".join(label_lines)
            ax.text(lon + 0.05, lat + 0.05, label_text,
                    transform=crs.PlateCarree(),
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    else:
        # Merged forecast at midpoint
        preds = df_step[df_step['Station'] == "merged"]
        if preds.empty:
            continue

        label_lines = ["merged", f"{mid_alt:.0f} m"]
        true_val = preds.iloc[0]['True']
        label_lines.append(f"True: {true_val:.1f}°C")

        for model in preds['Model'].unique():
            pred_val = preds[preds['Model'] == model]['Predicted'].iloc[0]
            label_lines.append(f"{model}: {pred_val:.1f}°C")

        label_text = "\n".join(label_lines)
        ax.text(mid_lon - 0.30, mid_lat - 0.20, label_text,
                transform=crs.PlateCarree(),
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.savefig(f"{output_dir}/forecast_step_{step}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"INFO: Plots saved to {output_dir}")
