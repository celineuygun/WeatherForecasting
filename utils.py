from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os, sys, re, subprocess, requests, time, glob, logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def find_latest_valid_gfs_cycle(current_time: datetime = None) -> tuple[str, datetime]:
    """
    Searches for the latest available GFS forecast cycle on the NOMADS server.

    This function checks today's and yesterday's possible GFS cycles (18z, 12z, 06z, 00z),
    and returns the latest one that is available (not in the future and accessible).

    Args:
        current_time (datetime, optional): Current time in UTC. If None, uses the system's current UTC time.

    Returns:
        tuple[str, datetime]: 
            - cycle (str): GFS cycle hour string (e.g. "00", "06", "12", "18").
            - cycle_datetime (datetime): Full datetime (UTC) representing the selected cycle.

    Exits:
        If no valid cycle is found, the script exits with an error message.
    """
    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
    current_time = current_time or datetime.now(timezone.utc)

    for delta_days in range(0, 2):  # today, yesterday
        check_date = current_time - timedelta(days=delta_days)
        ymd = check_date.strftime("%Y%m%d")
        for hour in (18, 12, 6, 0):
            cycle_dt = datetime(check_date.year, check_date.month, check_date.day, hour, tzinfo=timezone.utc)
            if cycle_dt > current_time:
                continue  # Skip future cycles
            cycle = f"{hour:02d}"
            url = f"{base_url}/gfs.{ymd}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f000"
            try:
                resp = requests.head(url, timeout=5)
                if resp.status_code == 200:
                    logging.info(f"Valid GFS cycle found: {ymd} {cycle}z")
                    return cycle, cycle_dt
            except Exception as e:
                logging.warning(f"Could not connect to {url}: {e}")
    
    logging.error("No valid GFS cycle found for today or yesterday.")
    sys.exit(1)


def gfs_download_worker(data):
    """
    Downloads a single GFS file with retry logic.

    Used by multiple threads to download GFS forecast files in parallel.

    Args:
        data (tuple): A tuple of:
            - url (str): The URL to the GFS file.
            - filepath (str): The local path to save the file.
            - hour (int): The forecast hour (used for logging/debugging).
    """
    url, filepath, hour = data
    retry = 0
    while retry < 3:
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 100_000:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    size = os.path.getsize(filepath)
                    if size < 100_000:
                        os.remove(filepath)
                        logging.warning(f"⚠️ File f{hour:03d} is too small ({size} bytes). Retrying...")
                        retry += 1
                        time.sleep(2)
                        continue
                    logging.info(f"f{hour:03d} downloaded ({size // 1024} KB)")
                else:
                    logging.error(f"Failed to fetch f{hour:03d} (HTTP {response.status_code})")
                    break
            else:
                logging.info(f"f{hour:03d} already exists and is valid.")
            break
        except Exception as e:
            logging.error(f"Exception downloading f{hour:03d}: {e}")
            retry += 1

def download_gfs(path: str, n_worker: int, start_date: datetime, forecast_time: int, increment: int, cycle_time: str, *_bbox_unused):
    """
    Downloads all required GFS forecast files for a given date and cycle.

    It saves the files into a folder named by date and verifies their integrity.

    Args:
        path (str): Path where files will be saved.
        n_worker (int): Number of threads for parallel downloads.
        start_date (datetime): The starting date and time of the forecast.
        forecast_time (int): Number of forecast hours.
        increment (int): Time step between forecast files.
        cycle_time (str): GFS cycle hour to download (e.g. "00", "06").
        *_bbox_unused: Unused arguments kept for compatibility.

    Exits:
        If forecast exceeds 384 hours, or downloaded files are missing/corrupt.
    """
    if forecast_time > 384:
        sys.exit("ERROR: GFS Downloader - Forecast time can't be more than 384")

    folder_path = f"{path}/{start_date.strftime('%Y-%m-%d')}"
    os.makedirs(folder_path, exist_ok=True)

    logging.info(f"GFS files will be saved in {folder_path}")

    base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
    ymd = start_date.strftime("%Y%m%d")
    forecast_hours = range(0, forecast_time + 1, increment)

    urls = [
        f"{base_url}/gfs.{ymd}/{cycle_time}/atmos/gfs.t{cycle_time}z.pgrb2.0p25.f{hour:03d}"
        for hour in forecast_hours
    ]
    paths = [
        f"{folder_path}/gfs_4_{ymd}_{cycle_time}00_{hour:03d}.grb2"
        for hour in forecast_hours
    ]

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        executor.map(gfs_download_worker, zip(urls, paths, forecast_hours))

    for filepath in paths:
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 100_000:
            logging.error(f"GFS file {Path(filepath).name} missing or corrupt after download.")
            sys.exit("Critical GFS file(s) are missing or invalid. Exiting.")

    logging.info(f"Full GFS download completed for cycle {cycle_time}")

def run_wps(wps_path: str, gfs_path: str, namelist_wps_path: str, max_dom: int, start_date: datetime, end_date: datetime, opts = None):
    """
    Executes the full WPS preprocessing workflow: geogrid.exe, ungrib.exe, metgrid.exe.

    Also updates the `namelist.wps` file with the correct start/end dates and domain settings.

    Args:
        wps_path (str): Path to the WPS directory.
        gfs_path (str): Path to the directory containing downloaded GFS files.
        namelist_wps_path (str): Path to the namelist.wps file to be modified.
        max_dom (int): Number of WRF domains.
        start_date (datetime): Forecast start datetime.
        end_date (datetime): Forecast end datetime.
        opts (dict, optional): Additional parameters to override in namelist.
    """
    wps_params = {
        "max_dom": str(max_dom),
        "start_date": start_date.strftime("%Y-%m-%d_%H:%M:%S"),
        "end_date": end_date.strftime("%Y-%m-%d_%H:%M:%S")
    }
    
    if opts:
        wps_params.update(opts)

    for key in ["parent_id", "parent_grid_ratio", "i_parent_start", "j_parent_start", "e_we", "e_sn"]:
        value = wps_params.get(key)
        if value != None and len(value.split(",")) != max_dom:
            sys.exit(f"ERROR: WPS - length of {key} value mismatched to max_dom parameter")

    with open(namelist_wps_path, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            for variable, value in wps_params.items():
                matched = re.search(rf"{variable}\s*=\s*[^,]+,", line)
                if matched:
                    index_of_equal_sign = line.find("=")

                    if variable in ["wrf_core", "map_proj", "geog_data_path", "out_format", "prefix", "fg_name"]:
                        lines[i] = f"{line[:index_of_equal_sign + 1]} '{value}',\n"
                        continue

                    if variable in ["start_date", "end_date", "geog_data_res"]:
                        formatted = f"'{value}',"
                        lines[i] = f"{line[:index_of_equal_sign + 1]} {formatted * max_dom}\n"
                        continue
                    
                    lines[i] = f"{line[:index_of_equal_sign + 1]} {str(value)},\n"

    with open(namelist_wps_path, "w") as file:
        file.writelines(lines)
    
    logging.info(f"INFO: WPS - Configuration file updated")

    # Delete FILE* and met_em* files from previous run
    subprocess.run([f"rm {wps_path}/FILE*"],shell=True)
    subprocess.run([f"rm {wps_path}/PFILE*"], shell=True)
    subprocess.run([f"rm {wps_path}/met_em*"], shell=True)
    subprocess.run([f"rm {wps_path}/GRIBFILE*"], shell=True)
    subprocess.run([f"rm {wps_path}/geo_em*"], shell=True)

    # Execute geogrid.exe
    subprocess.run("./geogrid.exe", cwd=wps_path)
    logging.info("INFO: WPS - geogrid.exe completed")

    # Create a link to GFS dataset
    subprocess.run(["./link_grib.csh", f"{gfs_path}/{start_date.strftime('%Y-%m-%d')}/*"], cwd=wps_path)
    logging.info("INFO: WPS - GFS dataset linked successfully")

    # Create a symlink to GFS Variable Table
    if os.path.exists(f"{wps_path}/Vtable"):
        logging.info("INFO: WPS - Vtable.GFS is already linked")
    else:
        subprocess.run(["ln", "-sf" ,f"{wps_path}/ungrib/Variable_Tables/Vtable.GFS", "Vtable"], cwd=wps_path)
        logging.info("INFO: WPS - Symlink of Vtable.GFS created")
    
    # Execute ungrib.exe
    subprocess.run("./ungrib.exe", cwd=wps_path)
    logging.info("INFO: WPS - ungrib.exe completed")

    # Execute metgrid.exe
    subprocess.run("./metgrid.exe", cwd=wps_path)
    logging.info("INFO: WPS - metgrid.exe completed")

    logging.info("INFO: WPS - Process completed. met_em files is ready")

def move_wrf_outputs(wrf_path: str, wrfout_path: str):
    """
    Organizes and moves WRF output files (wrfout/ wrfrst) into a folder structure by date and domain.

    Args:
        wrf_path (str): Path where WRF produced the output files.
        wrfout_path (str): Base directory to store and organize output files.
    """
    def get_wrf_output_path(base_output_dir: str, forecast_time: datetime) -> str:
        return os.path.join(base_output_dir, forecast_time.strftime("%Y-%m-%d"))

    all_outputs = sorted(
        glob.glob(os.path.join(wrf_path, "wrfout_d0*_*")) +
        glob.glob(os.path.join(wrf_path, "wrfrst_d0*_*"))
    )

    for file_path in all_outputs:
        file_name = os.path.basename(file_path)
        try:
            parts = file_name.split("_")
            domain_str = parts[1]
            if not domain_str.startswith("d"):
                logging.warning(f"Unexpected domain format in file: {file_name}")
                continue
            domain = domain_str[1:]

            timestamp_str = "_".join(parts[2:])
            forecast_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H:%M:%S")

            # Round down to nearest hour for folder
            forecast_time = forecast_time.replace(minute=0, second=0)

        except (IndexError, ValueError) as e:
            logging.warning(f"Skipping file with unexpected format: {file_name} ({e})")
            continue

        date_dir = get_wrf_output_path(wrfout_path, forecast_time)
        target_dir = os.path.join(date_dir, f"d{domain}")
        os.makedirs(target_dir, exist_ok=True)

        subprocess.run(["mv", file_name, target_dir], cwd=wrf_path)
        logging.info(f"Moved {file_name} → {target_dir}")

def run_wrf(wps_path: str, wrf_path: str, wrfout_path: str, namelist_input_path: str, run_days: int, max_dom: int, start_date: datetime, end_date: datetime, num_proc: int, wrfout_saved_domain: int, opts=None):
    """
    Runs the WRF model simulation end-to-end: prepares `namelist.input`, runs `real.exe` and `wrf.exe`,
    and saves outputs to structured folders.

    Args:
        wps_path (str): Path to WPS directory (for linking met_em* files).
        wrf_path (str): Directory where WRF executables are located.
        wrfout_path (str): Directory to store final WRF outputs.
        namelist_input_path (str): Path to the namelist.input file.
        run_days (int): Duration of the forecast in days.
        max_dom (int): Number of domains (e.g., 2 for d01 + d02).
        start_date (datetime): Start datetime of simulation.
        end_date (datetime): End datetime of simulation.
        num_proc (int): Number of processors (used with mpirun).
        wrfout_saved_domain (int): Domain number to save (2 for d02).
        opts (dict, optional): Dictionary of namelist overrides.

    Exits:
        If configuration is invalid or WRF execution fails.
    """

    wrf_params = {
        "run_days": str(run_days),
        "start_year": str(start_date.year),
        "start_month": f"{start_date.month:02d}",
        "start_day": f"{start_date.day:02d}",
        "start_hour": f"{start_date.hour:02d}",
        "end_year": str(end_date.year),
        "end_month": f"{end_date.month:02d}",
        "end_day": f"{end_date.day:02d}",
        "end_hour": f"{end_date.hour:02d}",
        "max_dom": str(max_dom),
        "e_we": "100,106",
        "e_sn": "100,106",
        "dx": "9000,9000",
        "dy": "9000,9000",
        "grid_id": "1,2",
        "parent_id": "1,1",
        "i_parent_start": "1,30",
        "j_parent_start": "1,30",
        "parent_grid_ratio": "1,3",
        "parent_time_step_ratio": "1,3"
    }

    if opts:
        wrf_params.update(opts)

    for key in ["e_we", "e_sn", "e_vert", "dx", "dy", "grid_id", "parent_id", "i_parent_start", "j_parent_start", "parent_grid_ratio", "parent_time_step_ratio"]:
        value = wrf_params.get(key)
        if value is not None and len(value.split(",")) != max_dom:
            sys.exit(f"ERROR: WRF Model - length of {key} value mismatched to max_dom parameter")

    if wrfout_saved_domain > max_dom:
        sys.exit("ERROR: WRF Model - Maximum saved WRF output file domain must be equal or lower to max_domain parameter")

    with open(namelist_input_path, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        for variable, value in wrf_params.items():
            if re.search(rf"{variable}\s*=\s*[^,]+,", line):
                index_of_equal_sign = line.find("=")
                if variable in ["start_year", "start_month", "start_day", "start_hour", "end_year", "end_month", "end_day", "end_hour"]:
                    lines[i] = f"{line[:index_of_equal_sign + 1]} {((value + ', ') * max_dom)}\n"
                else:
                    lines[i] = f"{line[:index_of_equal_sign + 1]} {value},\n"

    with open(namelist_input_path, "w") as file:
        file.writelines(lines)

    logging.info("INFO: WRF Model - Configuration file updated")
    logging.info(f"INFO: WRF Model - Model will take a simulation from {start_date.strftime('%Y-%m-%d_%H:%M:%S')} to {end_date.strftime('%Y-%m-%d_%H:%M:%S')}")

    subprocess.run([f"rm {wrf_path}/met_em*"], shell=True)
    subprocess.run([f"rm {wrf_path}/wrfout*"], shell=True)
    subprocess.run([f"rm {wrf_path}/wrfrst*"], shell=True)

    subprocess.run([f"ln -sf {wps_path}/met_em* ."], shell=True, cwd=wrf_path)
    logging.info("INFO: WRF Model - met_em* files has been linked")

    subprocess.run([f"mpirun -np {num_proc} ./real.exe"], shell=True, cwd=wrf_path)
    logging.info("INFO: WRF Model - real.exe executed")

    rsl_error = subprocess.check_output(["tail --lines 1 rsl.error.0000"], shell=True, cwd=wrf_path)
    if re.search("SUCCESS COMPLETE REAL_EM INIT", str(rsl_error)):
        subprocess.run([f"mpirun -np {num_proc} ./wrf.exe"], shell=True, cwd=wrf_path)
        logging.info("INFO: WRF Model - Simulation completed")
    else:
        sys.exit("ERROR: WRF Model - Check namelist.input configuration")

    move_wrf_outputs(wrf_path, wrfout_path)

    log_files = glob.glob(f"{wrf_path}/rsl.*")
    if log_files:
        first_file = os.path.basename(log_files[0])
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', first_file)
        if timestamp_match:
            first_time = datetime.strptime(timestamp_match.group(), "%Y-%m-%d_%H:%M:%S")
        else:
            first_time = start_date
        log_dir = f"{wrfout_path}/{first_time.strftime('%Y-%m-%d')}/logs"
        os.makedirs(log_dir, exist_ok=True)
        for log in log_files:
            subprocess.run([f"mv {log} {log_dir}/"], shell=True, cwd=wrf_path)
            logging.info(f"INFO: Moved {log} to {log_dir}")



    log_files = glob.glob(f"{wrf_path}/rsl.*")
    if log_files:
        first_file = os.path.basename(log_files[0])
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', first_file)
        if timestamp_match:
            first_time = datetime.strptime(timestamp_match.group(), "%Y-%m-%d_%H:%M:%S")
        else:
            first_time = start_date
        log_dir = f"{wrfout_path}/{first_time.strftime('%Y-%m-%d')}/logs"
        os.makedirs(log_dir, exist_ok=True)
        for log in log_files:
            subprocess.run([f"mv {log} {log_dir}/"], shell=True, cwd=wrf_path)
            logging.info(f"INFO: Moved {log} to {log_dir}")

def calculate_execution_time(start: float, stop: float):
    """
    Calculates and prints total execution time (in seconds, minutes, or hours).

    Args:
        start (float): Starting time (from `time.time()`).
        stop (float): Ending time.

    Exits:
        Always calls `sys.exit(0)` after printing the execution time.
    """
    if stop - start < 60:
        execution_duration = ("%1d" % (stop - start))
        logging.info(f"INFO: Automation - Process completed in {execution_duration} seconds")
    elif stop - start < 3600:
        execution_duration = ("%1d" % ((stop - start) / 60))
        logging.info(f"INFO: Automation - Process completed in {execution_duration} minutes")
    else:
        execution_duration = ("%1d" % ((stop - start) / 3600))
        logging.info(f"INFO: Automation - Process complete in {execution_duration} hours")
    sys.exit(0)