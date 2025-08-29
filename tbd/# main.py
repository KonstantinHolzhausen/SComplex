# main.py

import os
from config import EXPERIMENT_ID, DATA_DIR, PICKLE_DIR

# Data creation
from data_creation.create_signals import run_simulations
from data_creation.signal_logger import log_signals

# Data loading
from data_loading.load_csvs import load_simulation_data, summarize_simulations

# Data joining
from data_joining.join_dataframes import join_data, load_or_create_pickle

# Data cleaning
from data_cleaning.clean_data import clean_dataframe

# Metrics
from metrics.compute_metrics import compute_all_metrics

# Logging
from logging.experiment_logger import save_metrics_and_log

# Plotting
from plotting.persistence_diagrams import plot_persistence
from plotting.topology_summary import plot_topology_summary

def main():
    print(f"Starting pipeline for experiment: {EXPERIMENT_ID}")

    # Step 1: Create Data
    signals = run_simulations()
    log_signals(signals)

    # Step 2: Load Data
    raw_dataframes = load_simulation_data(DATA_DIR)
    summarize_simulations(raw_dataframes)

    # Step 3: Join Data
    joined_df = load_or_create_pickle(raw_dataframes, PICKLE_DIR, EXPERIMENT_ID)

    # Step 4: Clean Data
    cleaned_df = clean_dataframe(joined_df)

    # Step 5: Compute Metrics
    metrics_df = compute_all_metrics(cleaned_df)

    # Step 6: Save and Log
    save_metrics_and_log(metrics_df, EXPERIMENT_ID, PICKLE_DIR)

    # Step 7: Plotting
    plot_persistence(metrics_df, EXPERIMENT_ID)
    plot_topology_summary(metrics_df, cleaned_df)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
