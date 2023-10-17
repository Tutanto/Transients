"""
Transient Event Simulation Script

This Python script is designed to simulate transient astrophysical events using the Gammapy library and parallel processing. 
It generates a series of transient event simulations, varying parameters like spectral index, amplitude, and decay time. 
Each simulated event is saved with its associated model, and the entire set of simulations is managed in a structured directory.

The script utilizes various scientific libraries, such as NumPy, Astropy, and Gammapy, and custom utilities for efficient parallel processing. 
It also logs simulation details for analysis and records the execution time.

The key components of the script include:
- Defining the parameters for transient event simulations.
- Loading Instrument Response Functions (IRFs) for the simulations.
- Setting up the observation geometry and background model.
- Creating a pool of CPU processes for parallel execution.
- Running simulations in parallel for different parameter combinations.
- Saving the simulation results and associated models.
- Logging simulation details and recording execution time.

This script is useful for researchers and scientists working in astrophysics and gamma-ray astronomy to perform detailed simulations of transient astrophysical phenomena for further analysis.
"""

# Import necessary libraries
import time
import numpy as np
import pandas as pd
import astropy.units as u
from itertools import product
from astropy.time import Time
from astropy.coordinates import SkyCoord

# Import specific modules and classes from various packages
from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    ExpDecayTemporalModel,
    FoVBackgroundModel,
)

# Import the Pool class for parallel processing
from multiprocessing import Pool

# Import custom modules and variables
from GPyUtils.logging_config import logging_conf
from GPyUtils.wobbles import WobbleMaker
from modules.functions import simulate_transient
from modules.variables import *

# Start measuring the execution time
start_time = time.time()

if __name__ == '__main__':
    # Name of the output directory
    name_dir = "transient_parallel_test"

    # Set the number of CPU processes for parallel execution
    num_processes = 4  # Change this to the desired number of processes
    
    # Define the observation parameters
    single_livetime = 1 * u.hr
    tot_livetime = 5 * u.hr
    energy = [1, 300]

    # Create arrays of parameters for simulation
    idx = np.arange(2, 3, 0.5)
    ampl = np.logspace(-12, -7, 5)
    t0 = np.arange(1000, 3000, 500) * u.s
    
    # Create the output directory
    output_dir = simulated_data / name_dir
    output_dir.mkdir(parents=True)
    
    # Path to the Instrument Response Functions (IRFs)
    irf_path = astri_irf
    
    # Loading the IRFs
    irfs = load_irf_dict_from_file(irf_path)

    # Set up logging configuration
    logger = logging_conf(path_to_logs, "simulator.log")

    # Define the source position in galactic coordinates
    source_pos = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")    

    # Generate wobble positions for simulations
    Wb = WobbleMaker(source_pos, 0.4 * u.deg)
    wobble_positions = Wb.make_wobbles()

    # Define map geometry for binned simulation
    energy_reco = MapAxis.from_energy_bounds(
        energy[0], energy[1], unit="TeV", name="energy", nbin=10, per_decade=True
    )

    geom = WcsGeom.create(
        skydir=source_pos,
        binsz=0.02,
        width=(2, 2),
        frame="galactic",
        axes=[energy_reco],
    )
    
    # Create a separate energy axis for true energy
    energy_true = MapAxis.from_energy_bounds(
        round(energy[0] - energy[0] * 0.2), round(energy[1] + energy[1] * 0.2),
        unit="TeV", name="energy_true", nbin=20, per_decade=True
    )

    # Define a migration axis for the energy dispersion map
    migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

    # Create an empty MapDataset for simulation
    empty = MapDataset.create(
        geom, 
        name="dataset-simu", 
        energy_axis_true=energy_true,
        migra_axis=migra_axis,
    )

    # Define the background model
    bkg_model = FoVBackgroundModel(dataset_name="dataset-simu")

    # Set the parameters for the Power Law Spectral Model
    spectral_model_pwl = PowerLawSpectralModel(
        index=2, amplitude="1e-7 TeV-1 cm-2 s-1", reference="1 TeV"
    )

    # Set the point source spatial model
    spatial_model_point = PointSpatialModel(
        lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic"
    )

    # Set the reference time and the temporal model parameters
    t_ref = Time.now()
    expdecay_model = ExpDecayTemporalModel(t_ref=t_ref.mjd * u.d, t0=2800 * u.s)

    # Log simulation parameters
    logger.debug(f"Simulated time: {tot_livetime}")
    logger.debug(f"Source name: {name_dir}")
    logger.debug(f"Source position: {source_pos}")
    logger.debug(f"Simulated energy: {energy_reco}")
    logger.debug(f"Simulated geometry: {geom.wcs}")

    # Create a list of parameter combinations for simulation
    params_list = []

    for i, A, t in product(idx, ampl, t0):
        params_list.append([i, A, t])

    # Set the chunk size for parallel processing
    chunksize = len(params_list) // (num_processes * 2)  # Adjust the chunksize as needed
    
    logger.debug(f"Starting the simulator with {num_processes} CPUs")

    # Prepare arguments for the parallel execution
    args_list = [(idx, params_list[idx], expdecay_model, spectral_model_pwl, 
                  spatial_model_point, bkg_model, name_dir, empty, wobble_positions, 
                  tot_livetime, single_livetime, irfs, output_dir, logger) for idx in range(len(params_list))]
    
    # Execute simulations in parallel using multiple CPU processes
    with Pool(processes=num_processes) as pool:
        pool.map(simulate_transient, args_list, chunksize=chunksize)  # Specify the chunksize parameter
    
    logger.debug(f"End of the simulations")
    
    # Create a DataFrame from the parameter list and save it to a CSV file
    df = pd.DataFrame(params_list, columns=['index', 'amplitude', 't0'])
    df.to_csv(output_dir / 'dataframe.csv', index=False)  # Save the DataFrame to a CSV file
    
    logger.debug(f'Dataframe saved in: {output_dir}')

    # Stop the timer and calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    logger.debug("Execution time:", execution_time, "seconds")
