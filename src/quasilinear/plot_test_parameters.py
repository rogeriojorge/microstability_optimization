#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Determine the current script's directory
this_path = Path(__file__).parent.resolve()

# Read the CSV file into a DataFrame
csv_file = os.path.join(this_path, 'test_out_nfp4_QH_initial_ln1.0_lt3.0', 'test_out_nfp4_QH_initial_ln1.0_lt3.0.csv')
df = pd.read_csv(csv_file)

# Define the parameters that are being varied
varied_parameters = ['nphi', 'nperiod', 'nlambda', 'nstep', 'dt', 'negrid', 'ngauss', 'aky_min', 'aky_max', 'naky', 'vnewk']
parameter_legend  = [r'$n{\phi}$', r'$n_{\mathrm{periods}}$', r'$n_{\lambda}$', r'$n_{\Delta t}$', r'$\Delta t$', r'$n_E$', r'$n_{\mathrm{untrapped}}$', r'$ky_{\mathrm{min}}$', r'$ky_{\mathrm{max}}$', r'$n_{ky}$', r'$v_{\mathrm{newk}}$']

# Create a dictionary to map parameters to their corresponding factor for variation
parameter_factors = {
    'nphi': 2.0, 'nperiod': 2.0, 'nlambda': 2.0, 'nstep': 2.0,
    'dt': 0.5, 'negrid': 2.0, 'ngauss': 2.0, 'aky_min': 0.5,
    'aky_max': 2.0, 'naky': 2.0, 'vnewk': 0.5
}

# Define an extensive list of markers and colors
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', 'X', '8', 'd', '.', '+']
colors = plt.cm.tab20.colors

# Initialize x-axis labels and positions
x_labels = ['base case']
x_positions = [1]

# Base case
base_case = df.iloc[0]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(1, base_case['weighted_growth_rate'], marker=markers[0], s=200, color=colors[0], label='Base Case')

# Plot other cases with different markers and labels
j=2
for i, param in enumerate(varied_parameters):
    if param == 'aky_min' or param == 'aky_max':
        continue
    factor = parameter_factors[param]
    assert df.iloc[i+1][param] == base_case[param] * factor or df.iloc[i+1][param] == base_case[param] * factor-1
    marker = markers[i + 1]
    color = colors[i + 1]
    label = f'{"half " if factor < 1 else "double "} {parameter_legend[i]}'
    x_labels.append(label)
    ax.scatter(j, df.iloc[i+1]['weighted_growth_rate'], marker=marker, s=250, color=color, label=label)
    j += 1

# Set labels and legends
# ax.set_xlabel('Parameter Variation', fontsize=16)
ax.set_ylabel(r'$f_Q$ (quasilinear estimate)', fontsize=16)
ax.set_title('Convergence Scan', fontsize=16)
# ax.legend(loc='upper left', fontsize=12)

# Increase tick label font sizes
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# Add horizontal dotted lines for +/- 10% from the base case
ax.axhline(base_case['weighted_growth_rate']*1.1, linestyle='--', color='gray', label='+10% from Base Case')
ax.axhline(base_case['weighted_growth_rate']*0.9, linestyle='--', color='gray', label='-10% from Base Case')

# Add text labels for the lines
ax.text(4, base_case['weighted_growth_rate']*1.09, '+10% of Base Case', ha='right', va='bottom', size=12)
ax.text(4, base_case['weighted_growth_rate']*0.91, '-10% of Base Case', ha='right', va='top', size=12)

# Set x-axis labels and positions
ax.set_xticks(range(1, len(x_labels) + 1))
ax.set_xticklabels(x_labels, rotation=45, fontsize=12, ha='right')

# Save the plot as an image
plt.tight_layout()
plt.savefig('test_parameters.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
