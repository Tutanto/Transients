from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import PowerLawTemporalModel

gti_t0 = Time("2023-06-26")
t_0= 6000 * u.hr
#t_0_2= 6000 * u.min
time_range = [gti_t0, gti_t0 + 5 * u.hr]
pl_model = PowerLawTemporalModel(alpha=-2.0, t_ref=gti_t0.mjd * u.d, t0=t_0)
pl_model_2 = PowerLawTemporalModel(alpha=-2.5, t_ref=gti_t0.mjd * u.d, t0=t_0)

pl_model.plot(time_range, label=f"{t_0}")
pl_model_2.plot(time_range, label=f"2.5")
#plt.grid(which="both")
plt.legend()
plt.yscale("log")