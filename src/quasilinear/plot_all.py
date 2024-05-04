#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
import matplotlib
import warnings
import matplotlib.ticker as ticker
import matplotlib.cbook
import argparse
from configurations import CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()

wfQ_array = [100, 10, 1, 0.1, 0]
W7X_directory = '/Users/rogeriojorge/local/QI_Er_configs/QI_Er_configs/W7X'

prefix_save = 'optimization'
results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
optimizer = 'least_squares'

OUT_DIR_all = os.path.join(this_path,results_folder,config['output_dir'])
figures_directory = os.path.join(OUT_DIR_all, f'figures')
os.makedirs(figures_directory, exist_ok=True)

neo_epseff_array = []
neo_sradial_array = []
simple_time_array = []
simple_confpart_pass_array = []
simple_confpart_trap_array = []
for weight_optTurbulence in wfQ_array:
    OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}_wFQ{weight_optTurbulence:.1f}"
    output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
    OUT_DIR = os.path.join(OUT_DIR_all,OUT_DIR_APPENDIX)
    os.chdir(OUT_DIR)
    
    token = open('neo_out.final','r')
    linestoken=token.readlines()
    eps_eff=[]
    s_radial=[]
    for x in linestoken:
        s_radial.append(float(x.split()[0])/90)
        eps_eff.append(float(x.split()[1])**(2/3))
    token.close()
    s_radial = np.array(s_radial)
    eps_eff = np.array(eps_eff)
    s_radial = s_radial[np.argwhere(~np.isnan(eps_eff))[:,0]]
    eps_eff = eps_eff[np.argwhere(~np.isnan(eps_eff))[:,0]]
    neo_epseff_array.append(eps_eff)
    neo_sradial_array.append(s_radial)
    
    simple_time_array.append(np.loadtxt(f'time_array.txt'))
    simple_confpart_pass_array.append(np.loadtxt(f'confpart_pass_array.txt'))
    simple_confpart_trap_array.append(np.loadtxt(f'confpart_trap_array.txt'))

## Epsilon Effective Plot
os.chdir(W7X_directory)
token = open('neo_out.W7X','r')
linestoken=token.readlines()
eps_eff_W7X=[]
s_radial_W7X=[]
for x in linestoken:
    s_radial_W7X.append(float(x.split()[0])/150)
    eps_eff_W7X.append(float(x.split()[1])**(2/3))
token.close()
s_radial_W7X = np.array(s_radial_W7X)[np.argwhere(~np.isnan(np.array(eps_eff_W7X)))[:,0]]
eps_eff_W7X = np.array(eps_eff_W7X)[np.argwhere(~np.isnan(np.array(eps_eff_W7X)))[:,0]]

os.chdir(figures_directory)
fig = plt.figure(figsize=(7, 3), dpi=200)
ax = fig.add_subplot(111)
plt.plot(s_radial_W7X,eps_eff_W7X, '--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(neo_sradial_array[i],neo_epseff_array[i], label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
ax.set_yscale('log')
plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)
plt.xlim([0,1])
plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(figures_directory,f'neo_out_{config["output_dir"]}.pdf'), dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)

## Fast particle confinement plot
os.chdir(W7X_directory)
W7X_simple_time_array = np.loadtxt(f'time_array.txt')
W7X_simple_confpart_pass_array = np.loadtxt(f'confpart_pass_array.txt')
W7X_simple_confpart_trap_array = np.loadtxt(f'confpart_trap_array.txt')

os.chdir(figures_directory)
fig = plt.figure(figsize=(7, 3), dpi=200)
plt.semilogx(W7X_simple_time_array, 1 - (W7X_simple_confpart_pass_array + W7X_simple_confpart_trap_array), '--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.semilogx(simple_time_array[i], 1 - (simple_confpart_pass_array[i] + simple_confpart_trap_array[i]), label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlim([1e-6, 1e-2])
plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")
plt.legend()
plt.tight_layout()
print("Saving figure")
plt.savefig(f'simple_out_{config["output_dir"]}.pdf', dpi=200)