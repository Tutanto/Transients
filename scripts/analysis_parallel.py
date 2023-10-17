"""
Fit Transient Event Datasets

This Python script fits transient event datasets using the Gammapy library and parallel processing. It reads event files, loads pre-simulated models, fits datasets, calculates significance, and logs the results. The significance is computed as the difference in log-likelihood (TS) between the model fit and a null hypothesis.

The script is designed to analyze multiple transient event datasets in parallel, allowing for efficient processing.

Key Components:
- Loading necessary libraries, including Gammapy and custom utilities.
- Defining a function, `fit_datasets`, to fit a dataset, calculate significance, and log the results.
- Reading event files and creating a collection of datasets.
- Loading a pre-simulated model for the source and adding background models.
- Running the fit using Minuit optimizer and calculating the TS.
- Saving the fitted model to a YAML file and logging the results.
- Performing the same fitting process for null hypothesis (no source) and computing the significance.
- Running the entire process in parallel for multiple datasets and saving the results in a DataFrame.
- Measuring the execution time.

The script is useful for researchers and scientists working with transient astrophysical events to analyze their datasets and estimate the significance of the detected sources.

"""
# Import necessary libraries
import time
import pandas as pd
from multiprocessing import Pool

# Import Gammapy libraries and custom utilities
from GPyUtils.logging_config import logging_conf
from modules.variables import *
from modules.functions import fit_datasets

# Start measuring the execution time
start_time = time.time()

if __name__ == '__main__':
    # Define the number of CPU processes for parallel execution
    num_processes = 2  # Change this to the desired number of processes
    
    # Name of the output directory
    name_dir = "transient_parallel_test"
    simulated_dir = simulated_data / name_dir
    irf_path = astri_irf

    # Set up logging
    logger = logging_conf(path_to_logs, "fit.log")
    
    # Initialize a list to store significance results
    significance_list = []

    # List folders in the simulated directory
    logger.debug(f'Reading folders in {simulated_dir}')
    paths = list(simulated_dir.rglob("transient_*"))
    paths.sort()
    logger.debug(f'Folders found: {len(paths)}')

    chunksize = len(paths) // (num_processes * 2)  # Adjust the chunksize as needed

    logger.debug(f"Starting the script with {num_processes} CPUs")

    args_list = [(path, name_dir, logger) for path in paths]
    
    # Run the `fit_datasets` function in parallel
    with Pool(processes=num_processes) as pool:
        significance_list = pool.map(fit_datasets, args_list)
    
    logger.debug(f"End of the simulations")

    # Read the parameter DataFrame
    df = pd.read_csv(simulated_dir / 'dataframe.csv')
    
    # Add the calculated significance values
    df['significance'] = significance_list
    
    # Save the updated DataFrame to a CSV file
    df.to_csv(simulated_dir / 'dataframe_fitted.csv', index=False)
    logger.debug(f'Dataframe saved in: {simulated_dir}')

    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Log the execution time
    logger.debug("Execution time:", execution_time, "seconds")