import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

from gammapy.maps import Map
from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import Models, FoVBackgroundModel

from GPyUtils.logging_config import logging_conf
from modules.variables import *

# Start the timer
start_time = time.time()

def fit_datasets(dir):
    # let's read all the event files:
    events = list(dir.rglob("events*.fits"))

    datasets = Datasets()

    for ev in events:
        event = EventList.read(ev)
        name_event = ev.stem[-4:]
        # Reading the Dataset
        read_dataset = MapDataset.read(filename=dir / f"dataset_{name_event}.fits.gz")
        dataset = read_dataset.copy(name=f"dataset_{name_event}")
        geom = dataset.geoms["geom"]
        counts = Map.from_geom(geom)
        counts.fill_events(event)
        dataset.counts = counts
        datasets.append(dataset)


    simulated_model = Models.read(dir / "simulated_model.yaml")
    models_joint = Models()

    model_joint = simulated_model.select(name_dir)
    models_joint.append(model_joint[0])

    for dataset in datasets:
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        models_joint.append(bkg_model)

    datasets.models = models_joint
    
    logger.debug(f"Model used: {simulated_model}")

    # Run the fit
    fit = Fit(store_trace=True)
    minuit_opts = {"tol": 0.1, "strategy": 0}
    fit.backend = "minuit"
    fit.optimize_opts = minuit_opts
    result_fit = fit.run(datasets)
    L1 = result_fit.total_stat
    logger.info(result_fit)
    datasets.models.write(dir / "result_model.yaml", overwrite=True)
    logger.info(f"Fitted Model saved: {dir} / result_model.yaml")

    models_joint.remove(name_dir)
    datasets.models = models_joint
    # Run the fit
    null_fit = fit.run(datasets)
    L0 = null_fit.total_stat
    logger.info(null_fit)

    TS = L0 - L1
    significance = round(np.sqrt(TS), 1)
    return significance

if __name__ == '__main__':
    logger = logging_conf(path_to_logs, f"fit.log")
    params_list = []

    # Name of the output dir
    name_dir = "transient_parallel_test"
    simulated_dir = simulated_data / name_dir
    irf_path = astri_irf

    logger.debug(f'Reading folders in {simulated_dir}')
    paths = list(simulated_dir.rglob("transient_*"))
    paths.sort()
    logger.debug(f'Folders founded: {len(paths)}')

    num_processes = 2  # Change this to the desired number of processes
    chunksize = len(paths) // (num_processes * 2)  # Adjust the chunksize as needed

    logger.debug(f"Starting the simulator with {num_processes} CPUs")
    with Pool(processes=num_processes) as pool:
        significance_list = pool.map(fit_datasets, paths)
    logger.debug(f"End of the simulations")

    df = pd.read_csv(simulated_dir / 'dataframe.csv')
    df['significance'] = significance_list
    df.to_csv(simulated_dir / 'dataframe_fitted.csv', index=False)  # Save the DataFrame to a CSV file
    logger.debug(f'Dataframe saved in: {simulated_dir}')

    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the execution time
    logger.debug("Execution time:", execution_time, "seconds")
