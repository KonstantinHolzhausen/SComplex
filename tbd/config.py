
import os
from datetime import datetime

# === Experiment Configuration ===
EXPERIMENT_ID = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EXPERIMENT_DESCRIPTION = "Baseline simulation with default FMU parameters"

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PICKLE_DIR = os.path.join(BASE_DIR, "data", "pickled")
LOG_DIR = os.path.join(BASE_DIR, "logs")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")

# === Simulation Settings ===
RUN_SIMULATIONS = True  # Toggle to skip simulation step if already done
SAVE_PICKLES = True     # Save intermediate joined data
VERBOSE = True          # Print detailed logs

# === Metric Settings ===
METRICS_TO_COMPUTE = ["correlation", "distance_correlation", "hsic", "dtw"]

# === Plotting Settings ===
PLOT_PERSISTENCE = True
PLOT_TOPOLOGY = True

# === Utility ===
def ensure_directories():
    for path in [DATA_DIR, PICKLE_DIR, LOG_DIR, METRICS_DIR, PLOTS_DIR]:
        os.makedirs(path, exist_ok=True)

# Call this at the start of your main script
ensure_directories()
