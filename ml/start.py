import subprocess
import sys

def show_menu():
    """
    Displays the run mode selection menu for the user.
    Options include per-station processing, merged processing, and exit.
    """
    print("\nWeather Forecasting Pipeline")
    print("Select run mode:")
    print("  [1] Per Station")
    print("  [2] Merged (all stations together)")
    print("  [3] Exit")

def ask_run_mode():
    """
    Prompts the user to select a processing mode from the menu.

    Returns:
        bool: True if per-station mode is selected, False for merged mode.

    Exits:
        Program exits cleanly if the user selects option 3.
    """
    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        elif choice == "3":
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid input. Please select 1, 2, or 3.")

def run_script(script_name, per_station):
    """
    Runs a Python script using subprocess with the given per_station flag.

    Args:
        script_name (str): Name of the script to execute.
        per_station (bool): Whether to pass per-station processing flag.

    Prints:
        Execution status of the script (success or failure).
    """
    print(f"\n→ Running: {script_name}")
    command = ["python", script_name]
    if per_station:
        command.append("--per_station")

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        print(f"Error in {script_name}")
    else:
        print(f"Completed: {script_name}")

def run_pipeline(per_station):
    """
    Executes the full pipeline of preprocessing, training, forecasting, evaluating,
    fusion (if per-station), and visualization scripts in sequence.

    Args:
        per_station (bool): Flag to indicate whether the run is per-station or merged.
    """
    # Define script sequence depending on mode
    scripts = [
        "train.py",
        "forecast.py",
        "evaluate.py",
        "visualize.py"
    ]
    if per_station:
        scripts.insert(3, "fusion.py")  # Insert fusion before visualization

    for script in scripts:
        run_script(script, per_station)

    print("\nINFO: All steps completed successfully.")

if __name__ == "__main__":
    while True:
        show_menu()
        per_station = ask_run_mode()
        run_pipeline(per_station)
