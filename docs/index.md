# Weather Forecasting

This project combines a physics-based numerical weather prediction model (WRF) with a data-driven machine learning pipeline to generate short-term temperature forecasts using SYNOP observations.

## Running Forecast

To run the WRF-based physical forecast:
```bash
python corsica_forecast.py
```

This script:

* Orchestrates the full WRF simulation process (WPS + WRF)
* Downloads GFS data if necessary
* Prepares boundary and initial conditions
* Launches WRF over a predefined domain (Corsica)

## Visualization
```bash
python visualize.py
```
Generates forecast maps over Corsica using WRF output. Maps are saved to the plots/ directory.

## Machine Learning Workflow

The ml/ directory contains all components needed to train, test, and visualize machine learning models for temperature forecasting. These models operate either per station or on a merged dataset combining all stations.

| File           | Description |
|----------------|-------------|
| `start.py`     | Interactive CLI menu to launch the entire ML pipeline step-by-step |
| `train.py`     | Trains regression models (e.g., XGBoost, Random Forest) either per station or globally |
| `forecast.py`  | Generates temperature forecasts using trained models |
| `evaluate.py`  | Calculates standard error metrics like MAE, RMSE, R², EVS, MAPE |
| `fusion.py`    | Applies prediction fusion strategies (only in per-station mode) |
| `visualize.py` | Generates plots (residuals, rolling MAE, scatter, etc.) |
| `utils.py`     | Shared utility functions for evaluation and data handling |
| `models.py`    | Defines ML models like Random Forest, XGBoost, Neural Network |
| `preprocess.py`| Preprocesses and transforms SYNOP data into model-ready format |

### How to Run the ML Pipeline

There are two ways to run the pipeline: using an interactive menu or via command-line arguments.

### 1. Interactive Menu

Launch the interactive pipeline menu:

```bash
cd ml
python start.py
```

You will be prompted to select a run mode:

```
Weather Forecasting Pipeline
Select run mode:
  [1] Per Station
  [2] Merged (all stations together)
  [3] Exit
```

* Option 1 runs the entire pipeline separately for each weather station.
* Option 2 runs the pipeline once using the merged dataset.
* Option 3 exits the program.

Note: Fusion strategies are only applied when running in per-station mode.

### 2. Command-Line Arguments

Each step can be executed independently with (per-station mode) or without (merged mode) the `--per_station` flag.

```bash
# Training
python train.py --per_station   # per-station mode
python train.py                 # merged mode

# Forecasting
python forecast.py --per_station
python forecast.py

# Evaluation
python evaluate.py --per_station
python evaluate.py

# Fusion (per-station only, no need for a flag)
python fusion.py

# Visualization
python visualize.py --per_station
python visualize.py
```

### ML Outputs
After running the pipeline, several output directories and files are generated to store models, metrics, forecasts, visualizations, and evaluation results.

| Folder              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `ml/results/`           | Stores prediction CSVs, evaluation metrics, fusion results, and plots.     |
| `ml/results/per_station/` | Output for per-station mode. Contains one CSV per variable.              |
| `ml/results/merged/`    | Output for merged mode.                                                    |
| `ml/models/`            | Contains trained ML models in `.pkl` format        |
| `ml/plots/`             | Contains static maps showing forecasts on Corsica for each model and step. |



## References
- [WPS User Guide (UCAR)](https://mmg.atm.ucdavis.edu/wp-content/uploads/2014/10/WPS-Duda.pdf)
- [NOMADS – GFS Data Access](https://nomads.ncep.noaa.gov)
- [SYNOP API (data.corsica)](https://www.data.corsica/explore/dataset/observation-meteorologique-historiques-france-synop0/information)
