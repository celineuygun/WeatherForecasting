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
FORECAST_DAYS           = 1           # Number of forecast days
GFS_STEP_HOURS          = 3           # GFS temporal resolution (every 3 hours)
GFS_PARALLEL_DOWNLOADS  = 4           # Number of parallel downloads for GFS
WRF_PROCS               = 8           # Number of processors for WRF execution
WRF_OUTPUT_DOMAIN       = 2           # Domain to save output (d02)
MAX_DOM                 = 2           # Total number of nested domains

# GFS Bounding box for Corsica
GFS_BBOX = {
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
end_date = start_date + timedelta(days=FORECAST_DAYS)
print(f"Forecast period: {start_date} to {end_date}")

# Compute forecast duration in hours.
# Add one extra GFS step to ensure WRF has enough boundary data
forecast_hours = int((end_date - start_date).total_seconds() / 3600) + GFS_STEP_HOURS

# Download GFS data
utils.download_gfs(
    gfs_dir,
    GFS_PARALLEL_DOWNLOADS,
    start_date,
    forecast_hours,
    GFS_STEP_HOURS,
    gfs_cycle,
    GFS_BBOX["left"],
    GFS_BBOX["right"],
    GFS_BBOX["top"],
    GFS_BBOX["bottom"]
)

# Run WPS preprocessing steps
utils.run_wps(
    wps_dir,
    gfs_dir,
    namelist_wps,
    MAX_DOM,
    start_date,
    end_date
)

# Run WRF simulation
utils.run_wrf(
    wps_dir,
    wrf_dir,
    wrfout_dir,
    namelist_wrf,
    FORECAST_DAYS,
    MAX_DOM,
    start_date,
    end_date,
    WRF_PROCS,
    WRF_OUTPUT_DOMAIN
)

utils.calculate_execution_time(start_time, time.time())
print("Corsica forecast completed successfully!")
print("GFS data directory:", gfs_dir)
