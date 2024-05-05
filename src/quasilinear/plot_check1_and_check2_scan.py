#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import warnings
import argparse
from configurations import CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
parser.add_argument("--wfQ", type=float, default=10)
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
this_path = Path(__file__).parent.resolve()
prefix_scan_ln_lt = 'scan_ln_lt'
prefix_scan_s_alpha = 'scan_s_alpha'
prefix_save = 'optimization'
results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
weight_optTurbulence = args.wfQ
optimizer = 'least_squares'

### Fix some values for plotting several configurations together
plot_extent_fix_gamma = True
plot_extent_fix_weighted_gamma = True
plot_gamma_min = 0
plot_overall_gamma_min = 0
plot_weighted_gamma_min = 0
plot_overall_weighted_gamma_min = 0
plot_gamma_max = 0.5
plot_overall_gamma_max = 0.25
plot_weighted_gamma_max = 0.5
plot_overall_weighted_gamma_max = 0.3

# Define output directories and create them if they don't exist
this_path = Path(__file__).parent.resolve()

if config['output_dir']=='W7-X':
    OUT_DIR_APPENDIX=config['output_dir']
    OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'])
else:
    OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}"
    OUT_DIR_APPENDIX+=f'_wFQ{weight_optTurbulence:.1f}'
    OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],OUT_DIR_APPENDIX)
output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
figures_directory = os.path.join(OUT_DIR, f'figures')
os.makedirs(figures_directory, exist_ok=True)

# Define a function to plot and save the figure
def plot_and_save(save_name, data, xlabel, ylabel, clb_title, plotExtent=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if plotExtent is None:
        im = plt.imshow(data.T, cmap='jet', origin='lower', interpolation='hermite')
    else:
        im = plt.imshow(data.T, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
    clb = plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
    title = clb.ax.set_title(clb_title, fontsize=20, pad=10)
    if clb_title==r'$\gamma/\langle k_{\perp}^2 \rangle$' or clb_title==r'$\gamma_{\textrm{max}}$': # Adjust title position to the right
        title.set_position([1.4, 1.0])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_aspect('auto')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if clb_title==r'$\gamma_{\textrm{max}}$' and ylabel=='$a/L_T$':
        if plot_extent_fix_gamma: plt.clim(plot_gamma_min,plot_gamma_max)
    if clb_title==r'$\gamma_{\textrm{max}}$' and ylabel==r'$\alpha$':
        if plot_extent_fix_gamma: plt.clim(plot_overall_gamma_min,plot_overall_gamma_max)
    if clb_title==r'$\gamma/\langle k_{\perp}^2 \rangle$' and ylabel=='$a/L_T$':
        if plot_extent_fix_weighted_gamma: plt.clim(plot_weighted_gamma_min,plot_weighted_gamma_max)
    if clb_title==r'$\gamma/\langle k_{\perp}^2 \rangle$' and ylabel==r'$\alpha$':
        if plot_extent_fix_weighted_gamma: plt.clim(plot_overall_weighted_gamma_min,plot_overall_weighted_gamma_max)
    if ylabel==r'$\alpha$': # Convert y-tick labels to units of pi
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'))    
    plt.tight_layout()
    plt.savefig(save_name, format='pdf', bbox_inches='tight')

# Load and process the first CSV
csv_file_ln_lt = os.path.join(OUT_DIR,f'scan_ln_lt_{OUT_DIR_APPENDIX}.csv')
df = pd.read_csv(csv_file_ln_lt)
ln_values = df['ln'].unique()
lt_values = df['lt'].unique()
plotExtent = [0 * min(ln_values), max(ln_values), 0 * min(lt_values), max(lt_values)]
growth_rate_array = df.pivot_table(index='ln', columns='lt', values='growth_rate').values
omega_array = df.pivot_table(index='ln', columns='lt', values='omega').values
ky_array = df.pivot_table(index='ln', columns='lt', values='ky').values
weighted_growth_rate_array = df.pivot_table(index='ln', columns='lt', values='weighted_growth_rate').values

# Load and process the second CSV
csv_file_s_alpha = os.path.join(OUT_DIR,f'scan_s_alpha_{OUT_DIR_APPENDIX}.csv')
df_location = pd.read_csv(csv_file_s_alpha)
s_values = df_location['s'].unique()
alpha_values = df_location['alpha'].unique()
plotExtent_location = [min(s_values), max(s_values), min(alpha_values), max(alpha_values)]
growth_rate_location_array = df_location.pivot_table(index='s', columns='alpha', values='growth_rate').values
omega_location_array = df_location.pivot_table(index='s', columns='alpha', values='omega').values
ky_location_array = df_location.pivot_table(index='s', columns='alpha', values='ky').values
weighted_growth_rate_location_array = df_location.pivot_table(index='s', columns='alpha', values='weighted_growth_rate').values

# Plot and save for ln-lt
print('max gamma =', np.max(growth_rate_array), ', min gamma =', np.min(growth_rate_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_ln_lt}_gs2_scan_gamma.pdf'), growth_rate_array, '$a/L_n$', '$a/L_T$', r'$\gamma_{\textrm{max}}$', plotExtent=plotExtent)
print('max omega =', np.max(omega_array), ', min omega =', np.min(omega_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_ln_lt}_gs2_scan_omega.pdf'), omega_array, '$a/L_n$', '$a/L_T$', r'$\omega$', plotExtent=plotExtent)
print('max ky =', np.max(ky_array), ', min ky =', np.min(ky_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_ln_lt}_gs2_scan_ky.pdf'), ky_array, '$a/L_n$', '$a/L_T$', r'$k_y$', plotExtent=plotExtent)
print('max weighted gamma =', np.max(weighted_growth_rate_array), ', min weighted gamma =', np.min(weighted_growth_rate_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_ln_lt}_gs2_scan_weighted_gamma.pdf'), weighted_growth_rate_array, '$a/L_n$', '$a/L_T$', r'$\gamma/\langle k_{\perp}^2 \rangle$', plotExtent=plotExtent)

# Plot and save for s-alpha
print('max overall gamma =', np.max(growth_rate_location_array), ', min overall gamma =', np.min(growth_rate_location_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_s_alpha}_gs2_scan_gamma.pdf'), growth_rate_location_array, r'$s$', r'$\alpha$', r'$\gamma_{\textrm{max}}$', plotExtent=plotExtent_location)
print('max overall omega =', np.max(omega_location_array), ', min overall omega =', np.min(omega_location_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_s_alpha}_gs2_scan_omega.pdf'), omega_location_array, r'$s$', r'$\alpha$', r'$\omega$', plotExtent=plotExtent_location)
print('max overall ky =', np.max(ky_location_array), ', min overall ky =', np.min(ky_location_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_s_alpha}_gs2_scan_ky.pdf'), ky_location_array, r'$s$', r'$\alpha$', r'$k_y$', plotExtent=plotExtent_location)
print('max overall weighted gamma =', np.max(weighted_growth_rate_location_array), ', min overall weighted gamma =', np.min(weighted_growth_rate_location_array))
plot_and_save(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{prefix_scan_s_alpha}_gs2_scan_weighted_gamma.pdf'), weighted_growth_rate_location_array, r'$s$', r'$\alpha$', r'$\gamma/\langle k_{\perp}^2 \rangle$', plotExtent=plotExtent_location)