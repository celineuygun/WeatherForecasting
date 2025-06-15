
# Installation Guide — Hybrid Weather Forecasting System

This project sets up a hybrid weather forecasting pipeline that combines the **WRF model** (physics-based) with **Machine Learning** (data-driven) using local SYNOP observations. This guide explains how to install everything from scratch on a **Unix-like system** (macOS or Linux).

## Project Structure

```
WeatherForecasting/
├── docs/                   # Documentation
├── gfs_dataset/            # Downloaded GFS files
├── ml/                     # Machine learning training & evaluation
├── plots/                  # Forecast visualizations
├── validation_plots/       # Comparison with observations
├── wrf_output/             # WRF output (NetCDF)
├── WPS/                    # WPS tools + WPS_GEOG data inside
│   └── WPS_GEOG/           # Static geographic data
├── WRF/                    # WRF source and compiled binaries
├── corsica_forecast.py     # Main script to run WRF workflow
├── environment.yml         # Conda environment definition
├── utils.py                # Automation core functions
└── visualize.py            # Forecast map generation
```

## System Requirements

Install the following **system dependencies**:

### macOS (Homebrew)
```bash
brew install gcc gfortran make m4 wget mpich cmake netcdf
```

### Ubuntu
```bash
sudo apt update && sudo apt install -y \
  gfortran m4 build-essential wget \
  libopenmpi-dev openmpi-bin \
  netcdf-bin libnetcdf-dev libnetcdff-dev \
  python3-dev
```

## Conda Environment Setup

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed.

### Create Environment
```bash
conda env create -f environment.yml
conda activate wrf_project
```

## WRF and WPS Installation

### Download and Compile WRF
```bash
git clone https://github.com/wrf-model/WRF.git
cd WRF
./configure
./compile em_real >& compile.log
```

### Download and Compile WPS
```bash
cd ..
git clone https://github.com/wrf-model/WPS.git
cd WPS
./configure
./compile >& compile.log
```

### Install WPS_GEOG Data
```bash
cd WPS
mkdir WPS_GEOG
cd WPS_GEOG
wget http://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz
tar -xzf geog_high_res_mandatory.tar.gz
```

Make sure `namelist.wps` includes:
```fortran
geog_data_path = "/path/to/WPS_GEOG"
```

### Link GFS Variable Table
```bash
cd ..
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable
```
