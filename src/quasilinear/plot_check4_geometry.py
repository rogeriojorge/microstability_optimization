#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import matplotlib
import warnings
import matplotlib.ticker as ticker
import matplotlib.cbook
import argparse
from configurations import CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
parser.add_argument("--wfQ", type=float, default=10)
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()

prefix_save = 'optimization'
results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
weight_optTurbulence = args.wfQ
optimizer = 'least_squares'

s_radius = 0.25
alpha_fieldline = 0
phi_GS2 = np.linspace(-PARAMS['nperiod']*np.pi, PARAMS['nperiod']*np.pi, PARAMS['nphi'])

OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}"
OUT_DIR_APPENDIX+=f'_wFQ{weight_optTurbulence:.1f}'
output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],OUT_DIR_APPENDIX)
figures_directory = os.path.join(OUT_DIR, f'figures')
os.makedirs(figures_directory, exist_ok=True)

# START
vmec = Vmec(os.path.join(OUT_DIR, 'wout_final.nc'),verbose=False)
fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=True, show=False)
plt.savefig(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_geometry_profiles_s{s_radius}_alpha{alpha_fieldline}.pdf'))
plt.close()

matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
data = [fl1.grad_alpha_dot_grad_alpha[0,0,:], fl1.grad_s_dot_grad_s[0,0,:], fl1.modB[0,0,:], fl1.B_cross_grad_B_dot_grad_alpha[0,0,:]]
parameters = [r'$|\nabla \alpha|^2$', r'$|\nabla \psi|^2$', r'$|B|$', r'(\textbf B $\times \nabla B) \cdot \nabla \alpha$']
save_names = ['gyy','gxx','modB','BcrossgradBdotgradalpha']
phi = fl1.phi[0,0,:]

for parameter, d, save_name in zip(parameters, data, save_names):
    fig, ax = plt.subplots(figsize=(5, 6))
    plt.plot(phi, d,linewidth=2)
    plt.xlabel('Standard toroidal angle $\phi$', fontsize=18)
    plt.ylabel(parameter, fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory,f'{config["output_dir"]}_wFQ{weight_optTurbulence:.1f}_{save_name}_s{s_radius}_alpha{alpha_fieldline}.pdf'))
    plt.close()