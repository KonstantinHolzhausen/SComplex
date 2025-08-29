# main.py

import os
from config import EXPERIMENT_ID, DATA_DIR, PICKLE_DIR
import pickle


# Data creation
#from data_processor import run_simulations
#from data_processor import log_signals

from data_processor.data_processor import load_csvs
from data_processor.data_processor import join_dataframes, load_or_create_pickle
from data_processor.data_processor import clean_dataframe

# Metrics
from metrics import get_metrics
# Logging
from logging import save_metrics_and_log
# Plotting
#from plotting.persistence_diagrams import plot_persistence
#from plotting.topology_summary import plot_topology_summary

def main():
    print(f"Starting pipeline for experiment: {EXPERIMENT_ID}")

    # Step 1: Create Data
    #signals = run_simulations()
    #log_signals(signals)

    # Step 2: Load Data
    raw_dataframes = load_csvs(DATA_DIR)
    summarize_simulations(raw_dataframes) ##plotting
    joined_df = join_dataframes(raw_dataframes, PICKLE_DIR, EXPERIMENT_ID)  # Step 3: Join Data
    cleaned_df = clean_dataframe(joined_df)  # Step 4: Clean Data

    # Step 5: Compute Metrics
    if os.path.exists(PICKLE_DIR):
        if self.verbose:
            print(f"Loading joined dataframe from pickle: {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    metrics_df = compute_all_metrics(cleaned_df)
    save_metrics_and_log(metrics_df, EXPERIMENT_ID, PICKLE_DIR)

    # Step 7: Plotting
    plot_persistence(metrics_df, EXPERIMENT_ID)
    plot_topology_summary(metrics_df, cleaned_df)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
