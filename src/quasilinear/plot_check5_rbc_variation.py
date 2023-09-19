#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import warnings
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=-2)
args = parser.parse_args()
GS2_EXECUTABLE = '/Users/rogeriojorge/local/gs2/bin/gs2'
CONFIG = {
    -3: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/input.nfp1_QI',
        "output_dir": 'nfp1_QI_initial',
        "params": { 'nphi': 69,'nlambda': 21,'nperiod': 2.0,'nstep': 220,'dt': 0.5,
                    'aky_min': 0.3,'aky_max': 4.0,'naky': 8,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    },
    -2: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/input.nfp4_QH',
        "output_dir": 'nfp4_QH_initial',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    },
    -1: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/input.nfp2_QA',
        "output_dir": 'nfp2_QA_initial',
        "params": { 'nphi': 89,'nlambda': 25,'nperiod': 3.0,'nstep': 270,'dt': 0.4,
                    'aky_min': 0.4,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    }
}
prefix_save = 'rbc_variation'
results_folder = 'results'
figures_directory = 'figures'
config = CONFIG[args.type]
PARAMS = config['params']
OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],figures_directory)

OUTPUT_DIR_CSV = os.path.join(this_path,results_folder,config['output_dir'],f"{prefix_save}_{config['output_dir']}")
output_path_parameters_opt = os.path.join(OUTPUT_DIR_CSV,f'opt_dofs_loss_{prefix_save}_{config["output_dir"]}.csv')
output_path_parameters_scan = os.path.join(OUTPUT_DIR_CSV,f'scan_dofs_{prefix_save}_{config["output_dir"]}.csv')
output_path_parameters_min = os.path.join(OUTPUT_DIR_CSV,f'min_dofs_{prefix_save}_{config["output_dir"]}.csv')

vmec = Vmec(config['vmec_file'], verbose=False)
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")

df_scan = pd.read_csv(output_path_parameters_scan, delimiter=';')

# Find parameter changing instead of
# from check5_rbc_variation import vmec_index_scan_opt
# def transform_string(s): # transform vmec.dof_names into RBC and ZBS
#     sub_str = s.split(":")[1]
#     if 'rc' in sub_str:
#         sub_str = sub_str.replace('rc', 'RBC')
#     elif 'zs' in sub_str:
#         sub_str = sub_str.replace('zs', 'ZBS')
#     numbers = sub_str[sub_str.index("(")+1:sub_str.index(")")].split(',')
#     transposed_numbers = f"({numbers[1]},{numbers[0]})"
#     final_str = sub_str[:sub_str.index("(")] + transposed_numbers
#     return final_str
# vmec_index_scan_opt = 0
# parameter_changing = transform_string(vmec.dof_names[vmec_index_scan_opt])
rbc_columns = [col for col in df_scan.columns if col.startswith("RBC")]
zbs_columns = [col for col in df_scan.columns if col.startswith("ZBS")]
rbc_std = df_scan[rbc_columns].std()
zbs_std = df_scan[zbs_columns].std()
changing_rbc_columns = rbc_std[rbc_std != 0].index.tolist()
changing_zbs_columns = zbs_std[zbs_std != 0].index.tolist()
if changing_rbc_columns:
    parameter_changing = changing_rbc_columns
else:
    parameter_changing = changing_zbs_columns

df_scan = df_scan.sort_values(by=parameter_changing)
min_bound = np.min(df_scan[parameter_changing])
max_bound = np.max(df_scan[parameter_changing])

points_scan = np.linspace(min_bound,max_bound,len(df_scan[parameter_changing]))
fig = plt.figure();plt.plot(df_scan[parameter_changing], df_scan['growth_rate'], label='Scan')
plt.ylabel('Microstability Cost Function');plt.xlabel(parameter_changing);plt.legend();plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_growth_rate_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio')
plt.ylabel('Aspect ratio');plt.xlabel(parameter_changing);plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_aspect_ratio_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)')
plt.ylabel('Rotational Transform (1/q)');plt.xlabel(parameter_changing);plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_iota_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function')
plt.ylabel('Quasisymmetry cost function');plt.xlabel(parameter_changing);plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_quasisymmetry_scan.pdf'))
fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well')
plt.ylabel('Magnetic well');plt.xlabel(parameter_changing);plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_magnetic_well_scan.pdf'))
# fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
# plt.ylabel('Effective time');plt.xlabel(parameter_changing);plt.savefig('effective_1o_time_scan.pdf')

fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot(111, label="1")
ax2=fig.add_subplot(111, label="2", frame_on=False)
ax.set_xlabel('$RBC_{0,1}$', fontsize=20)
ax.tick_params(axis='x', labelsize=14)
line1, = ax.plot(df_scan[parameter_changing], df_scan['growth_rate'], color="C0", label='$f_Q$')
ax.set_ylabel("Microstability", color="C0", fontsize=20)
ax.tick_params(axis='y', colors="C0", labelsize=15)
ax.set_xlim((min_bound,max_bound))
ax.autoscale(enable=None, axis="y", tight=False)
line2, = ax2.plot(df_scan[parameter_changing], df_scan['quasisymmetry'], color="C1", label='$f_{QS}$')
ax2.yaxis.tick_right()
ax2.set_xticks([])
ax2.set_ylabel('Quasisymmetry', color="C1", fontsize=20) 
ax2.yaxis.set_label_position('right') 
ax2.tick_params(axis='y', colors="C1", labelsize=15)
ax2.set_xlim((min_bound,max_bound))
ax2.autoscale(enable=None, axis="y", tight=False)
plt.legend(handles=[line1, line2], prop={'size': 15})
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_quasisymmetry_vs_growthrate.pdf'))

# try:
#     df_opt = pd.read_csv(output_path_parameters_opt)
#     fig, ax = plt.subplots()
#     plt.plot(df_scan[parameter_changing], df_scan['growth_rate'], label='Scan')
#     ln, = ax.plot([], [], 'ro', markersize=1)
#     vl = ax.axvline(0, ls='-', color='r', lw=1)
#     patches = [ln,vl]
#     ax.set_xlim(min_bound,max_bound)
#     ax.set_ylim(np.min(0.8*df_scan['growth_rate']), np.max(df_scan['growth_rate']))
#     def update(frame):
#         ind_of_frame = df_opt.index[df_opt[parameter_changing] == frame][0]
#         df_subset = df_opt.head(ind_of_frame+1)
#         xdata = df_subset[parameter_changing]
#         ydata = df_subset['growth_rate']
#         vl.set_xdata([frame,frame])
#         ln.set_data(xdata, ydata)
#         return patches
#     ani = FuncAnimation(fig, update, frames=df_opt[parameter_changing])
#     ani.save(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_opt_animation.gif'), writer='imagemagick', fps=5)

#     fig = plt.figure()
#     plt.plot(df_opt[parameter_changing], df_opt['growth_rate'], 'ro', markersize=1, label='Optimizer')
#     plt.plot(df_scan[parameter_changing], df_scan['growth_rate'], label='Scan')
#     plt.ylabel('Microstability Cost Function');plt.xlabel(parameter_changing);plt.legend()
#     plt.savefig(os.path.join(OUT_DIR,f'{prefix_save}_{config["output_dir"]}_growth_rate_over_opt_scan.pdf'))
# except Exception as e: print(e)