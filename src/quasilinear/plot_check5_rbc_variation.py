#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import warnings
import argparse
from configurations import CONFIG
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
parser.add_argument("--wfQ", type=float, default=10)
args = parser.parse_args()
vmec_index_scan_opt = 0

results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
weight_optTurbulence = args.wfQ
optimizer = 'least_squares'
prefix_save = 'optimization'

OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}"
OUT_DIR_APPENDIX+=f'_wFQ{weight_optTurbulence:.1f}'
output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

figures_directory = os.path.join(OUT_DIR, f'figures')
os.makedirs(figures_directory, exist_ok=True)

output_path_parameters_scan = f'scan_dofs_{prefix_save}_{config["output_dir"]}.csv'
df_scan = pd.read_csv(output_path_parameters_scan, delimiter=';')

vmec = Vmec(os.path.join(OUT_DIR, 'input.final'),verbose=False)
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=4, nmin=-4, nmax=4, fixed=False)
surf.fix("rc(0,0)")
initial_dof = vmec.x[vmec_index_scan_opt]

rbc_columns = [col for col in df_scan.columns if col.startswith("RBC")]
zbs_columns = [col for col in df_scan.columns if col.startswith("ZBS")]
rbc_std = df_scan[rbc_columns].std()
zbs_std = df_scan[zbs_columns].std()
changing_rbc_columns = rbc_std[rbc_std != 0].index.tolist()
changing_zbs_columns = zbs_std[zbs_std != 0].index.tolist()
if changing_rbc_columns:
    parameter_changing = changing_rbc_columns[0]
else:
    parameter_changing = changing_zbs_columns[0]

df_scan = df_scan.sort_values(by=parameter_changing)
min_bound = np.min(df_scan[parameter_changing])
max_bound = np.max(df_scan[parameter_changing])

def transform_string(input_string, delimiter_left='(', delimiter_right=')'):
    # Split the string at the first occurrence of the left delimiter
    split_string = input_string.split(delimiter_left, 1)
    
    # Extract the prefix and the numbers
    prefix = split_string[0]
    numbers = split_string[1].strip(delimiter_right).split(',')
    
    # Reverse the order of the numbers
    numbers_reversed = numbers[::-1]
    
    # Format the string with the reversed numbers
    new_parameter = '{}_{{{},{}}}'.format(prefix, numbers_reversed[0], numbers_reversed[1])

    return new_parameter

axis_label = f'${transform_string(parameter_changing)}$'

points_scan = np.linspace(min_bound,max_bound,len(df_scan[parameter_changing]))
fig = plt.figure();plt.plot(df_scan[parameter_changing], df_scan['growth_rate'], label='Scan');plt.axvline(x=initial_dof, color='k', linestyle='--')
plt.ylabel('Microstability Cost Function');plt.xlabel(axis_label);plt.legend();plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_growth_rate_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio');plt.axvline(x=initial_dof, color='k', linestyle='--')
plt.ylabel('Aspect ratio');plt.xlabel(axis_label);plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_aspect_ratio_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)');plt.axvline(x=initial_dof, color='k', linestyle='--')
plt.ylabel('Rotational Transform (1/q)');plt.xlabel(axis_label);plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_iota_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function');plt.axvline(x=initial_dof, color='k', linestyle='--')
plt.ylabel('Quasisymmetry cost function');plt.xlabel(axis_label);plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_quasisymmetry_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well');plt.axvline(x=initial_dof, color='k', linestyle='--')
plt.ylabel('Magnetic well');plt.xlabel(axis_label);plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_magnetic_well_scan.pdf'))
# fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
# plt.ylabel('Effective time');plt.xlabel(parameter_changing);plt.savefig('effective_1o_time_scan.pdf')

fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax.set_xlabel(axis_label, fontsize=20)
ax.tick_params(axis='x', labelsize=14)
line1, = ax.plot(df_scan[parameter_changing], df_scan['growth_rate'], color="C0", label='$f_Q$')
ax.set_ylabel("Microstability", color="C0", fontsize=20)
ax.tick_params(axis='y', colors="C0", labelsize=15)
ax.set_xlim((min_bound,max_bound))
ax.autoscale(enable=None, axis="y", tight=False)
line2, = ax2.plot(df_scan[parameter_changing], df_scan['quasisymmetry'], color="C1", label='$f_{QS}$')
ax2.yaxis.tick_right()
plt.axvline(x=initial_dof, color='k', linestyle='--')
ax2.set_xticks([])
ax2.set_ylabel('Quasisymmetry', color="C1", fontsize=20) 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='y', colors="C1", labelsize=15)
ax2.set_xlim((min_bound,max_bound))
ax2.autoscale(enable=None, axis="y", tight=False)
plt.legend(handles=[line1, line2], prop={'size': 15})
plt.tight_layout()
plt.savefig(os.path.join(figures_directory,f'{prefix_save}_{config["output_dir"]}_quasisymmetry_vs_growthrate.pdf'))
