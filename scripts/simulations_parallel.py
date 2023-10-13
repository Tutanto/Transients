import time
import pandas as pd
import astropy.units as u
from itertools import product
from astropy.time import Time
from astropy.coordinates import SkyCoord

from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    ExpDecayTemporalModel,
    FoVBackgroundModel,
    SkyModel,
)

from multiprocessing import Pool

from GPyUtils.general_utils import simulate_parallel
from GPyUtils.logging_config import logging_conf
from GPyUtils.wobbles import WobbleMaker
from modules.variables import *

# Start the timer
start_time = time.time()

def simulate_transient(params):
    i, A, t, counter = params

    trans_id = "{0:04d}".format(counter)
    transient_dir = output_dir / f"transient_{trans_id}"
    transient_dir.mkdir(parents=True)
    expdecay_model.t0.value = t.value

    spectral_model_pwl.index.value = i
    spectral_model_pwl.amplitude.value = A

    sky_model_pntpwl = SkyModel(
        spectral_model=spectral_model_pwl,
        spatial_model=spatial_model_point,
        temporal_model=expdecay_model,
        name=name_dir,
    )

    models = Models([sky_model_pntpwl, bkg_model])
    models.write(transient_dir / "simulated_model.yaml", overwrite=True)

    simulate_parallel(empty, models, wobble_positions, tot_livetime, single_livetime, irfs, transient_dir, logger)

if __name__ == '__main__':

    # Name of the output dir
    name_dir = "transient_parallel_test"
    output_dir = simulated_data / name_dir
    output_dir.mkdir(parents=True)
    irf_path = astri_irf
    # Loading IRFs
    irfs = load_irf_dict_from_file(irf_path)

    # Set up log configurator
    logger = logging_conf(path_to_logs, "simulator.log")

    # Define the observation parameters (typically the observation duration and energy range):
    single_livetime = 1 * u.hr
    tot_livetime = 5 * u.hr
    energy = [1, 300]

    source_pos = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")    
    Wb = WobbleMaker(source_pos, 0.4*u.deg)
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
    # It is usually useful to have a separate binning for the true energy axis
    energy_true = MapAxis.from_energy_bounds(
        round(energy[0]-energy[0]*.2), round(energy[1]+energy[1]*.2), unit="TeV", name="energy_true", nbin=20, per_decade=True
    )

    #This provides the migration axis for the energy dispersion map. If not set, an EDispKernelMap is produced instead. Default is None
    migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")

    empty = MapDataset.create(
        geom, 
        name="dataset-simu", 
        energy_axis_true=energy_true,
        migra_axis=migra_axis,
        )

    # Define sky model to simulate the data.
    bkg_model = FoVBackgroundModel(dataset_name="dataset-simu")

    spectral_model_pwl = PowerLawSpectralModel(
        index=2, amplitude="1e-7 TeV-1 cm-2 s-1", reference="1 TeV"
    )
    spatial_model_point = PointSpatialModel(
        lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic"
    )

    t_ref = Time.now()
    expdecay_model = ExpDecayTemporalModel(t_ref=t_ref.mjd * u.d, t0=2800 * u.s)

    logger.debug(f"Simulated time: {tot_livetime}")
    logger.debug(f"Source name: {name_dir}")
    logger.debug(f"Source position: {source_pos}")
    logger.debug(f"Simulated energy: {energy_reco}")
    logger.debug(f"Simulated geometry: {geom.wcs}")

    '''idx = np.arange(2,3.5,.5)
    ampl = np.logspace(-12, -7, 10)
    t0 = np.arange(1000, 3000, 200) * u.s
    '''
    idx = [2., 2.5]
    ampl = [1.00000000e-12, 1.00000000e-11]
    t0 = [1000, 2000] * u.s
    params_list = []
    counter = 0

    for i, A, t in product(idx, ampl, t0):
        params_list.append([i, A, t, counter])
        counter+=1
                
    num_processes = 3  # Change this to the desired number of processes
    chunksize = len(params_list) // (num_processes * 2)  # Adjust the chunksize as needed
    
    logger.debug(f"Starting the simulator with {num_processes} CPUs")
    with Pool(processes=num_processes) as pool:
        pool.map(simulate_transient, params_list, chunksize=chunksize)  # Specify the chunksize parameter
    logger.debug(f"End of the simulations")
    for sublist in params_list:
        sublist.pop()
    df = pd.DataFrame(params_list, columns=['index', 'amplitude', 't0'])
    df.to_csv(output_dir / 'dataframe.csv', index=False)  # Save the DataFrame to a CSV file
    logger.debug(f'Dataframe saved in: {output_dir}')

    # End the timer
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the execution time
    logger.debug("Execution time:", execution_time, "seconds")
