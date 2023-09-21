#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import warnings
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)

CONFIG = {
    -3: {"name": 'nfp1_QI_initial', 'label_factor': 0.10},
    -2: {"name": 'nfp4_QH_initial', 'label_factor': 0.10},
    -1: {"name": 'nfp2_QA_initial', 'label_factor': 0.10},
     1: {"name": 'nfp2_QA', 'label_factor': 0.10},
     2: {"name": 'nfp4_QH', 'label_factor': 0.10}
}
same_max_gamma_gamma_overallgamma = False

ln = 1.0
lt = 3.0
folder_name = 'test_convergence'
results_directory = 'results'

parameters_to_plot = ['growth_rate','weighted_growth_rate']
label_parameters   = [r'max($\gamma$)', r'$\sum \gamma/\langle k_{\perp}^2 \rangle$']

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=-2)
parser.add_argument("--wfQ", type=float, default=0.0)
args = parser.parse_args()
config = CONFIG[args.type]['name']

# Define output directories and create them if they don't exist
this_path = Path(__file__).parent.resolve()
main_dir = os.path.join(this_path, results_directory, config)
out_dir = os.path.join(main_dir, f'{config}_wFQ{args.wfQ:.3f}_figures')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Read the CSV file into a DataFrame
csv_file = os.path.join(main_dir, f'{folder_name}_{config}_wFQ{args.wfQ:.3f}', f'{folder_name}_{config}.csv')
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
markers = ['o', 'X']#, 'D', '^', 'v', '<', '>', 'p', '*', 'H', 's', '8', 'd', '.', '+']
colors = plt.cm.tab20.colors
colors = [(r*0.9, g*0.9, b*0.9) for r, g, b in colors]

# Base case
base_case = df.iloc[0]

# Calculate max_gamma_plot using the filtered data
filtered_df_mode = df[(df['aky_min'] == df['aky_min'].mode().iloc[0]) & (df['aky_max'] == df['aky_max'].mode().iloc[0])]
max_gamma_plot = max(max(max(filtered_df_mode['growth_rate'].max(), filtered_df_mode['weighted_growth_rate'].max()), base_case['growth_rate']*1.15), base_case['weighted_growth_rate']*1.15)
min_gamma_plot = min(min(min(filtered_df_mode['growth_rate'].min(), filtered_df_mode['weighted_growth_rate'].min()), base_case['growth_rate']*0.85), base_case['weighted_growth_rate']*0.85)


for nn, parameter in enumerate(parameters_to_plot):
    # Initialize x-axis labels and positions
    x_labels = ['base case']
    x_positions = [1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(1, base_case[parameter], marker=markers[0], s=200, color=colors[0], label='Base Case')

    # Plot other cases with different markers and labels
    j=2
    for i, param in enumerate(varied_parameters):
        if param == 'aky_min' or param == 'aky_max':
            continue
        factor = parameter_factors[param]
        if param == 'nphi':
            index = df[df[param] == base_case[param] * factor - 1].index.to_list()[0]
            # print('i=',i,'j=',j,'factor=',factor,'index=',index,'param=',param,'df.iloc[index][param]=',df.iloc[index][param],'base_case[param]=',base_case[param],'base_case[param]*factor-1=',base_case[param]*factor-1)
        else:
            index = df[df[param] == base_case[param] * factor].index.to_list()[0]
            # print('i=',i,'j=',j,'factor=',factor,'index=',index,'param=',param,'df.iloc[index][param]=',df.iloc[index][param],'base_case[param]=',base_case[param],'base_case[param]*factor=',base_case[param]*factor)
        assert df.iloc[index][param] == base_case[param] * factor or df.iloc[index][param] == base_case[param] * factor-1
        marker = markers[1]
        color = colors[j-1]
        label = f'{"half " if factor < 1 else "double "} {parameter_legend[i]}'
        x_labels.append(label)
        ax.scatter(j, df.iloc[index][parameter], marker=marker, s=250, color=color, label=label)
        j += 1

    # Set labels and legends
    # ax.set_xlabel('Parameter Variation', fontsize=16)
    ax.set_ylabel(label_parameters[nn], fontsize=16)
    ax.set_title('Convergence Scan', fontsize=16)
    # ax.legend(loc='upper left', fontsize=12)

    # Increase tick label font sizes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add horizontal dotted lines for +/- 10% from the base case
    ax.axhline(base_case[parameter]*(1+CONFIG[args.type]['label_factor']), linestyle='--', color='gray', label='+10% from Base Case')
    ax.axhline(base_case[parameter]*(1-CONFIG[args.type]['label_factor']), linestyle='--', color='gray', label='-10% from Base Case')
    ax.axhline(base_case[parameter]*1.0, linestyle='--', color='gray', label='Base Case')

    # Add text labels for the lines
    ax.text(4, base_case[parameter]*1.085, r'+10\% of Base Case', ha='right', va='bottom', size=12)
    ax.text(4, base_case[parameter]*0.915, r'-10\% of Base Case', ha='right', va='top', size=12)

    # Set x-axis labels and positions
    ax.set_xticks(range(1, len(x_labels) + 1))
    xticklabels = ax.set_xticklabels(x_labels, rotation=45, fontsize=14, ha='right')
    for i, label in enumerate(xticklabels):
        label.set_color(colors[i])

    if (parameter == 'growth_rate' or parameter == 'weighted_growth_rate') and same_max_gamma_gamma_overallgamma:
        ax.set_ylim(min_gamma_plot, max_gamma_plot)

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{config}_test_{parameter}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()