import os
from datetime import datetime, timedelta, timezone
import time
import utils

# Path configuration
base_dir        = os.path.dirname(os.path.abspath(__file__))
gfs_dir         = os.path.join(base_dir, "gfs_dataset")
wps_dir         = os.path.join(base_dir, "WPS")
wrf_dir         = os.path.join(base_dir, "WRF", "test", "em_real")
wrfout_dir      = os.path.join(base_dir, "wrf_output")
namelist_wps    = os.path.join(wps_dir, "namelist.wps")
namelist_wrf    = os.path.join(wrf_dir, "namelist.input")

# Simulation configuration 
forecast_days           = 1           # Number of forecast days
gfs_step_hours          = 3           # GFS temporal resolution (every 3 hours)
gfs_parallel_downloads  = 4           # Number of parallel downloads for GFS
wrf_procs               = 8           # Number of processors for WRF execution
wrf_output_domain       = 2           # Domain to save output (d02)
max_dom                 = 2           # Total number of nested domains

# GFS Bounding box for Corsica
gfs_bbox = {
    "left": 6.0,
    "right": 11.0,
    "top": 44.5,
    "bottom": 41.0
}

# Start measuring time
start_time = time.time()

# Get latest valid GFS cycle and corresponding start date
gfs_cycle, start_date = utils.find_latest_valid_gfs_cycle()

# Define end date and forecast window
end_date = start_date + timedelta(days=forecast_days)
print(f"Forecast period: {start_date} to {end_date}")

# Compute forecast duration in hours.
# Add one extra GFS step to ensure WRF has enough boundary data
forecast_hours = int((end_date - start_date).total_seconds() / 3600) + gfs_step_hours

# Download GFS data
utils.download_gfs(
    gfs_dir,
    gfs_parallel_downloads,
    start_date,
    forecast_hours,
    gfs_step_hours,
    gfs_cycle,
    gfs_bbox["left"],
    gfs_bbox["right"],
    gfs_bbox["top"],
    gfs_bbox["bottom"]
)

# Run WPS preprocessing steps
utils.run_wps(
    wps_dir,
    gfs_dir,
    namelist_wps,
    max_dom,
    start_date,
    end_date
)

# Run WRF simulation
utils.run_wrf(
    wps_dir,
    wrf_dir,
    wrfout_dir,
    namelist_wrf,
    forecast_days,
    max_dom,
    start_date,
    end_date,
    wrf_procs,
    wrf_output_domain
)

utils.calculate_execution_time(start_time, time.time())
print("Corsica forecast completed successfully!")
print("GFS data directory:", gfs_dir)
