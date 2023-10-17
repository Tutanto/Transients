# Script Description:
# This script demonstrates the use of the Astropy and Gammapy libraries to create and plot PowerLawTemporalModel instances.
# It generates two PowerLawTemporalModel instances with different parameters and plots them using Matplotlib.

# Import necessary libraries
from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import PowerLawTemporalModel

# Define the reference time
gti_t0 = Time.now()

# Define time intervals and power law indices
t_a = 1000 * u.s
t_b = 2000 * u.s
alpha_a = -2.0
alpha_b = -2.5

# Define the time range for plotting
time_range = [gti_t0, gti_t0 + 5 * u.hr]

# Create PowerLawTemporalModel instances with different parameters
pl_model_a = PowerLawTemporalModel(alpha=alpha_a, t_ref=gti_t0.mjd * u.d, t0=t_a)
pl_model_b = PowerLawTemporalModel(alpha=alpha_b, t_ref=gti_t0.mjd * u.d, t0=t_b)

# Plot the PowerLawTemporalModels
pl_model_a.plot(time_range, label=f"t={t_a} - alpha={alpha_a}")
pl_model_b.plot(time_range, label=f"t={t_b} - alpha={alpha_b}")

# Add grid lines to the plot
plt.grid(which="both")

# Add a legend to the plot
plt.legend()

# Set the y-axis scale to logarithmic
plt.yscale("log")

# Display the plot
plt.show()