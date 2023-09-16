import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('test_out_nfp4_QH_initial_ln1.0_lt3.0/test_out_nfp4_QH_initial_ln1.0_lt3.0.csv')

# Define the parameters that are being varied
varied_parameters = ['nphi', 'nperiod', 'nlambda', 'nstep', 'dt', 'negrid', 'ngauss', 'aky_min', 'aky_max', 'naky']

# Create a dictionary to map parameters to their corresponding factor for variation
parameter_factors = {
    'nphi': 2.0, 'nperiod': 2.0, 'nlambda': 2.0, 'nstep': 2.0,
    'dt': 0.5, 'negrid': 2.0, 'ngauss': 2.0, 'aky_min': 0.5, 'aky_max': 2.0, 'naky': 2.0
}

# Create the plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))

# Define an extensive list of markers and colors
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', 'X', '8', 'd', '.', '+']
colors = plt.cm.tab20.colors

# Base case
base_case = df.iloc[0]
ax.scatter(base_case['weighted_growth_rate'], base_case['weighted_growth_rate'], marker=markers[0], s=200, color=colors[0], label='Base Case')

# Plot other cases with different markers and labels
for i, param in enumerate(varied_parameters):
    factor = parameter_factors[param]
    sub_df = df[df[param] == base_case[param] * factor]
    marker = markers[i + 1]
    color = colors[i + 1]
    label = f'{"" if factor < 1 else "double "} {param}'
    ax.scatter(sub_df['weighted_growth_rate'], sub_df['weighted_growth_rate'], marker=marker, s=200, color=color, label=label)

# Set labels and legends
ax.set_xlabel('Parameter Variation', fontsize=16)
ax.set_ylabel('Weighted Growth Rate', fontsize=16)
ax.legend(loc='upper left', fontsize=12)

# Increase tick label font sizes
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Set aspect ratio to equal
plt.gca().set_aspect('equal')

# Save the plot as an image
plt.tight_layout()
plt.savefig('test_parameters.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
