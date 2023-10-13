import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.variables import *

# Name of the output dir
name_dir = "transient_parallel_test"
simulated_dir = simulated_data / name_dir

# Read the CSV file
df = pd.read_csv(simulated_dir / 'dataframe_fitted.csv')

# Convert the 't0' column to numeric values
df['t0'] = df['t0'].str.replace(' s', '').astype(float)

# Generate a heatmap for each index value
for index_value in df['index'].unique():
    subset = df[df['index'] == index_value]

    # Pivot the DataFrame to have amplitude as rows, t0 as columns, and significance as values
    pivot_df = subset.pivot(index='amplitude', columns='t0', values='significance')

    # Create the heatmap
    plt.figure()
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', cbar=True)
    plt.title(f'Heatmap for Index {index_value}')
    plt.xlabel('t0')
    plt.ylabel('amplitude')
    plt.show()