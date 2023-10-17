import numpy as np
from gammapy.maps import Map
from gammapy.modeling import Fit
from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.modeling.models import (
    Models, 
    SkyModel,
    FoVBackgroundModel
    )
from GPyUtils.general_utils import simulate_parallel

# Define a function to simulate a transient event
def simulate_transient(args):

    counter, params, expdecay_model, spectral_model_pwl, spatial_model_point, bkg_model, name_dir, empty, wobble_positions, tot_livetime, single_livetime, irfs, output_dir, logger = args

    i, A, t = params

    # Create a unique ID for this transient event
    trans_id = "{0:04d}".format(counter)
    
    # Create a directory to store simulation results for this event
    transient_dir = output_dir / f"transient_{trans_id}"
    transient_dir.mkdir(parents=True)
    
    # Set the decay time of the exponential temporal model
    expdecay_model.t0.value = t.value

    # Set the spectral parameters for the Power Law Spectral Model
    spectral_model_pwl.index.value = i
    spectral_model_pwl.amplitude.value = A

    # Create a SkyModel for the transient event
    sky_model_pntpwl = SkyModel(
        spectral_model=spectral_model_pwl,
        spatial_model=spatial_model_point,
        temporal_model=expdecay_model,
        name=name_dir,
    )

    # Combine the transient event's SkyModel with the background model
    models = Models([sky_model_pntpwl, bkg_model])
    
    # Write the model to a YAML file for later analysis
    models.write(transient_dir / "simulated_model.yaml", overwrite=True)

    # Run the simulation for this transient event in parallel
    simulate_parallel(empty, models, wobble_positions, tot_livetime, single_livetime, irfs, transient_dir, logger)

def fit_datasets(args):

    dir, name_dir, logger = args
    
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

