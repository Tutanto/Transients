# Script Description:
# This script reads data from a CSV file, performs data preprocessing, 
# and generates heatmaps for different index values.
# The 'amplitude' column is formatted to display values 
# in scientific notation with 2 significant digits.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.variables import *
from GPyUtils.general_utils import save_figure

# Name of the directory with analysis results
name_dir = "transient_parallel_test"
simulated_dir = simulated_data / name_dir

# Read the CSV file containing the data
df = pd.read_csv(simulated_dir / 'dataframe_fitted.csv')

# Convert the 't0' column to numeric values
df['t0'] = df['t0'].str.replace(' s', '').astype(float)

# Custom function to format values to 2 significant digits in scientific notation
def format_scientific(value, precision=2):
    formatted_value = f'{value:.{precision}e}'
    return float(formatted_value)  # Convert back to float

# Apply the formatting function to the 'amplitude' column
df['amplitude'] = df['amplitude'].apply(format_scientific)

# Define the custom ordering for the 'amplitude' column
custom_order = df['amplitude'].sort_values().unique()

# Create a directory for saving the generated figures
path_to_figures.mkdir(parents=True, exist_ok=True)

# Generate a heatmap for each unique 'index' value
for index_value in df['index'].unique():
    subset = df[df['index'] == index_value]

    # Pivot the DataFrame to have 'amplitude' as rows, 't0' as columns, and 'significance' as values
    pivot_df = subset.pivot(index='amplitude', columns='t0', values='significance')

    # Create and save the heatmap figure
    plt.figure()
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', cbar=True)
    plt.title(f'Heatmap for Index {index_value}')
    plt.xlabel('t0')
    plt.ylabel('amplitude')

    # Save the figure with a filename based on the index value
    save_figure(path_to_figures / f'map_{index_value}.jpg')