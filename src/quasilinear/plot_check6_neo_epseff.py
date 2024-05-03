#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import shutil
import subprocess
from pathlib import Path
from simsopt.mhd import Vmec, Boozer
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

if not os.path.isfile("boozmn_final.nc"):
    booz_in_template = os.path.join(this_path, '..', 'util', 'in_booz.final')
    booz_input_file = f'in_booz.final'
    shutil.copyfile(booz_in_template,booz_input_file)
    boozxform_executable = '/Users/rogeriojorge/bin/xbooz_xform'

    bashCommand = f'{boozxform_executable} {booz_input_file}'
    run_booz = subprocess.Popen(bashCommand.split())
    run_booz.wait()

if not os.path.isfile("neo_out.final"):
    neo_template = os.path.join(this_path, '..', 'util', 'neo_in.final')
    neo_input_file = f'neo_in.final'
    shutil.copyfile(neo_template,neo_input_file)
    neo_executable = '/Users/rogeriojorge/bin/xneo'
    bashCommand = f'{neo_executable} final'
    run_neo = subprocess.Popen(bashCommand.split())
    run_neo.wait()

token = open('neo_out.final','r')
linestoken=token.readlines()
eps_eff=[]
s_radial=[]
for x in linestoken:
    s_radial.append(float(x.split()[0])/100)
    eps_eff.append(float(x.split()[1])**(2/3))
token.close()
s_radial = np.array(s_radial)
eps_eff = np.array(eps_eff)
s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
fig = plt.figure(figsize=(7, 3), dpi=200)
ax = fig.add_subplot(111)
plt.plot(s_radial,eps_eff, label=f'eps eff wFQ{weight_optTurbulence:.1f}')
ax.set_yscale('log')
plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)
plt.xlim([0,1])

plt.tight_layout()
fig.savefig(os.path.join(figures_directory,f'neo_out_{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}.pdf'), dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)