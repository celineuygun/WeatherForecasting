import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from wrf import getvar, to_np, latlon_coords, smooth2d
import cartopy.crs as crs
import cartopy.feature as cfeature
from datetime import datetime
import os
import glob

# Path configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.path.join(base_dir, "wrf_output")
plot_dir = os.path.join(base_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# User input for forecast date and domain
dates = sorted([d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))])
if not dates:
    print("No forecast dates found in wrf_output.")
    exit()

print("Available forecast dates:")
for i, d in enumerate(dates):
    print(f"{i + 1}. {d}")
selected_date = dates[int(input("Select a date by number: ")) - 1]

domains = sorted([d for d in os.listdir(os.path.join(output_root, selected_date)) if d.startswith("d")])
print("\nAvailable domains:")
for i, d in enumerate(domains):
    print(f"{i + 1}. {d}")
selected_domain = domains[int(input("Select a domain by number: ")) - 1]

domain_path = os.path.join(output_root, selected_date, selected_domain)
files = sorted(glob.glob(f"{domain_path}/wrfout_{selected_domain}_*"))
if not files:
    print("No WRF output files found.")
    exit()

print("\nAvailable forecast times:")
for i, f in enumerate(files):
    timestamp = "_".join(f.split("_")[-2:])
    print(f"{i + 1}. {timestamp}")
selected_file = files[int(input("Select a time by number: ")) - 1]

# Prepare timestamp for plot title
timestamp = "_".join(selected_file.split("_")[-2:])
dt = datetime.strptime(timestamp, "%Y-%m-%d_%H:%M:%S")
timestamp_label = dt.strftime("%Y-%m-%d %H:%M UTC")
filename_time = dt.strftime("%Y-%m-%d_%H-%M")

print(f"\nOpening file: {selected_file}")
ncfile = Dataset(selected_file)

def setup_map(ax):
    """
    Configure a Cartopy map projection with geographic features over Corsica.

    Parameters:
        ax (GeoAxes): The matplotlib axis object to configure.

    Returns:
        GeoAxes: The modified axis with coastlines and borders.
    """
    ax.set_extent([7.8, 10.2, 41.1, 43.1], crs=crs.PlateCarree())
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    return ax

def plot_t2(nc, lats, lons):
    """
    Plot 2-meter temperature field from a WRF NetCDF file with Ajaccio and Bastia STATIONS marked and labeled with temperatures and altitude.

    Parameters:
        nc (Dataset): NetCDF dataset.
        lats (ndarray): Latitude coordinates.
        lons (ndarray): Longitude coordinates.

    Saves:
        PNG file in the 'plots' directory.
    """
    t2 = getvar(nc, "T2", timeidx=0) - 273.15  # Kelvin to Celsius
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=crs.PlateCarree())
    setup_map(ax)

    c = ax.contourf(to_np(lons), to_np(lats), to_np(t2), 20, transform=crs.PlateCarree(), cmap="RdBu_r")
    plt.colorbar(c, orientation="horizontal", pad=0.05, label="2m Temperature (°C)")

    STATIONS = {
        "Ajaccio": {
            "lat": 41.918,
            "lon": 8.792667,
            "alt": 5
        },
        "Bastia": {
            "lat": 42.540667,
            "lon": 9.485167,
            "alt": 10
        }
    }

    lat_vals = to_np(lats)
    lon_vals = to_np(lons)
    t2_vals = to_np(t2)

    for name, info in STATIONS.items():
        lat_s, lon_s, alt = info["lat"], info["lon"], info["alt"]

        dist_sq = (lat_vals - lat_s) ** 2 + (lon_vals - lon_s) ** 2
        iy, ix = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        temp_val = t2_vals[iy, ix]

        ax.plot(lon_s, lat_s, marker='o', color='black', markersize=6, transform=crs.PlateCarree())
        ax.text(lon_s + 0.05, lat_s + 0.05,
                f"{name}\n{temp_val:.1f}°C\n{alt} m",
                transform=crs.PlateCarree(),
                fontsize=9, weight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title(f"2-Meter Temperature (°C)\n{timestamp_label}")
    fig.savefig(f"{plot_dir}/t2_{selected_domain}_{filename_time}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wind(nc, lats, lons):
    """
    Plot 10-meter wind speed derived from U10 and V10 fields.

    Parameters:
        nc (Dataset): NetCDF dataset.
        lats (ndarray): Latitude coordinates.
        lons (ndarray): Longitude coordinates.

    Saves:
        PNG file in the 'plots' directory.
    """
    u10 = getvar(nc, "U10", timeidx=0)
    v10 = getvar(nc, "V10", timeidx=0)
    wspd = ((u10**2 + v10**2)**0.5)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=crs.PlateCarree())
    setup_map(ax)
    c = ax.contourf(to_np(lons), to_np(lats), to_np(wspd), 20, transform=crs.PlateCarree(), cmap="viridis")
    plt.colorbar(c, orientation="horizontal", pad=0.05, label="10m Wind Speed (m/s)")
    plt.title(f"10m Wind Speed (m/s)\n{timestamp_label}")
    fig.savefig(f"{plot_dir}/wind_{selected_domain}_{filename_time}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_mslp(nc, lats, lons):
    """
    Plot Mean Sea Level Pressure (MSLP) from WRF output.

    Parameters:
        nc (Dataset): NetCDF dataset.
        lats (ndarray): Latitude coordinates.
        lons (ndarray): Longitude coordinates.

    Saves:
        PNG file in the 'plots' directory.
    """
    slp = getvar(nc, "slp", timeidx=0)
    slp_interp = smooth2d(slp, 3)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=crs.PlateCarree())
    setup_map(ax)
    c = ax.contour(to_np(lons), to_np(lats), to_np(slp_interp), 10, transform=crs.PlateCarree(), colors="black")
    plt.clabel(c, inline=1, fontsize=10)
    plt.title(f"Mean Sea Level Pressure (hPa)\n{timestamp_label}")
    fig.savefig(f"{plot_dir}/mslp_{selected_domain}_{filename_time}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# Prepare coordinates for plotting
t2 = getvar(ncfile, "T2", timeidx=0)
lats, lons = latlon_coords(t2)

# Plot selection menu
print("\nAvailable plots:")
plot_options = {
    "1": ("2m Temperature", plot_t2),
    "2": ("10m Wind Speed", plot_wind),
    "3": ("Mean Sea Level Pressure", plot_mslp),
}
for key, (label, _) in plot_options.items():
    print(f"{key}. {label}")

selected = input("Select plot(s) by number (e.g., 1 2 3): ").split()
for s in selected:
    if s in plot_options:
        print(f"Generating: {plot_options[s][0]}")
        plot_options[s][1](ncfile, lats, lons)

print(f"Plots saved in: {plot_dir}")
