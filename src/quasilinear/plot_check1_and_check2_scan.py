#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import warnings
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)

config = 'nfp4_QH_initial'

prefix_scan_ln_lt = 'scan_ln_lt'
prefix_scan_s_alpha = 'scan_s_alpha'
results_folder = 'results'

### Fix some values for plotting several configurations together
plot_extent_fix_gamma = False
plot_extent_fix_weighted_gamma = False
plot_gamma_min = 0
plot_overall_gamma_min = 0
plot_weighted_gamma_min = 0
plot_overall_weighted_gamma_min = 0
if 'QA' in config:
    plot_gamma_max = 0.41
    plot_overall_gamma_max = 0.16
    plot_weighted_gamma_max = 0.33
    plot_overall_weighted_gamma_max = 0.17
else:
    plot_gamma_max = 0.41
    plot_overall_gamma_max = 0.16
    plot_weighted_gamma_max = 0.33
    plot_overall_weighted_gamma_max = 0.17

# Define output directories and create them if they don't exist
this_path = Path(__file__).parent.resolve()
figures_directory = os.path.join(this_path, results_folder, config, 'figures')
out_dir = os.path.join(this_path, figures_directory)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Define a function to plot and save the figure
def plot_and_save(save_name, data, xlabel, ylabel, clb_title, plotExtent=None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if plotExtent is None:
        im = plt.imshow(data.T, cmap='jet', origin='lower', interpolation='hermite')
    else:
        im = plt.imshow(data.T, cmap='jet', extent=plotExtent, origin='lower', interpolation='hermite')
    clb = plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
    title = clb.ax.set_title(clb_title, fontsize=20, pad=10)
    if clb_title==r'$\sum\gamma/\langle k_{\perp}^2 \rangle$': # Adjust title position to the right
        title.set_position([1.6, 1.0])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_aspect('auto')
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if clb_title==r'max($\gamma$)' and ylabel=='$a/L_T$':
        if plot_extent_fix_gamma: plt.clim(plot_gamma_min,plot_gamma_max)
    if clb_title==r'max($\gamma$)' and ylabel==r'$\alpha$':
        if plot_extent_fix_gamma: plt.clim(plot_overall_gamma_min,plot_overall_gamma_max)
    if clb_title==r'$\sum\gamma/\langle k_{\perp}^2 \rangle$' and ylabel=='$a/L_T$':
        if plot_extent_fix_weighted_gamma: plt.clim(plot_weighted_gamma_min,plot_weighted_gamma_max)
    if clb_title==r'$\sum\gamma/\langle k_{\perp}^2 \rangle$' and ylabel==r'$\alpha$':
        if plot_extent_fix_weighted_gamma: plt.clim(plot_overall_weighted_gamma_min,plot_overall_weighted_gamma_max)
    if ylabel==r'$\alpha$': # Convert y-tick labels to units of pi
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'))    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, save_name), format='pdf', bbox_inches='tight')

# Load and process the first CSV
OUT_DIR_ln_lt = os.path.join(this_path,results_folder,config,f'{prefix_scan_ln_lt}_{config}')
csv_file_ln_lt = os.path.join(OUT_DIR_ln_lt,f'{prefix_scan_ln_lt}_{config}.csv')
df = pd.read_csv(csv_file_ln_lt)
ln_values = df['ln'].unique()
lt_values = df['lt'].unique()
plotExtent = [0 * min(ln_values), max(ln_values), 0 * min(lt_values), max(lt_values)]
growth_rate_array = df.pivot(index='ln', columns='lt', values='growth_rate').values
omega_array = df.pivot(index='ln', columns='lt', values='omega').values
ky_array = df.pivot(index='ln', columns='lt', values='ky').values
weighted_growth_rate_array = df.pivot(index='ln', columns='lt', values='weighted_growth_rate').values

# Load and process the second CSV
OUT_DIR_s_alpha = os.path.join(this_path,results_folder,config,f'{prefix_scan_s_alpha}_{config}')
csv_file_s_alpha = os.path.join(OUT_DIR_s_alpha,f'{prefix_scan_s_alpha}_{config}.csv')
df_location = pd.read_csv(csv_file_s_alpha)
s_values = df_location['s'].unique()
alpha_values = df_location['alpha'].unique()
plotExtent_location = [min(s_values), max(s_values), min(alpha_values), max(alpha_values)]
growth_rate_location_array = df_location.pivot(index='s', columns='alpha', values='growth_rate').values
omega_location_array = df_location.pivot(index='s', columns='alpha', values='omega').values
ky_location_array = df_location.pivot(index='s', columns='alpha', values='ky').values
weighted_growth_rate_location_array = df_location.pivot(index='s', columns='alpha', values='weighted_growth_rate').values

# Plot and save for ln-lt
print('max gamma =', np.max(growth_rate_array), ', min gamma =', np.min(growth_rate_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_ln_lt}_gs2_scan_gamma.pdf'), growth_rate_array, '$a/L_n$', '$a/L_T$', r'max($\gamma$)', plotExtent=plotExtent)
print('max omega =', np.max(omega_array), ', min omega =', np.min(omega_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_ln_lt}_gs2_scan_omega.pdf'), omega_array, '$a/L_n$', '$a/L_T$', r'$\omega$', plotExtent=plotExtent)
print('max ky =', np.max(ky_array), ', min ky =', np.min(ky_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_ln_lt}_gs2_scan_ky.pdf'), ky_array, '$a/L_n$', '$a/L_T$', r'$k_y$', plotExtent=plotExtent)
print('max weighted gamma =', np.max(weighted_growth_rate_array), ', min weighted gamma =', np.min(weighted_growth_rate_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_ln_lt}_gs2_scan_weighted_gamma.pdf'), weighted_growth_rate_array, '$a/L_n$', '$a/L_T$', r'$\sum\gamma/\langle k_{\perp}^2 \rangle$', plotExtent=plotExtent)

# Plot and save for s-alpha
print('max overall gamma =', np.max(growth_rate_location_array), ', min overall gamma =', np.min(growth_rate_location_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_s_alpha}_gs2_scan_gamma.pdf'), growth_rate_location_array, r'$s$', r'$\alpha$', r'max($\gamma$)', plotExtent=plotExtent_location)
print('max overall omega =', np.max(omega_location_array), ', min overall omega =', np.min(omega_location_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_s_alpha}_gs2_scan_omega.pdf'), omega_location_array, r'$s$', r'$\alpha$', r'$\omega$', plotExtent=plotExtent_location)
print('max overall ky =', np.max(ky_location_array), ', min overall ky =', np.min(ky_location_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_s_alpha}_gs2_scan_ky.pdf'), ky_location_array, r'$s$', r'$\alpha$', r'$k_y$', plotExtent=plotExtent_location)
print('max overall weighted gamma =', np.max(weighted_growth_rate_location_array), ', min overall weighted gamma =', np.min(weighted_growth_rate_location_array))
plot_and_save(os.path.join(figures_directory,f'{config}_{prefix_scan_s_alpha}_gs2_scan_weighted_gamma.pdf'), weighted_growth_rate_location_array, r'$s$', r'$\alpha$', r'$\sum\gamma/\langle k_{\perp}^2 \rangle$', plotExtent=plotExtent_location)