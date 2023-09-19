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
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=0)
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()
prefix_save = 'geometry'
results_folder = 'results'
figures_directory = 'figures'
if args.type == 0:
    vmec_file = os.path.join(this_path, '..', 'vmec_inputs', 'wout_nfp4_QH.nc')
    config = 'nfp4_QH_initial'
elif args.type == 1:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp2_QA_QA_onlyQS/wout_final.nc')
    config = 'nfp2_QA_QA_onlyQS'
elif args.type == 2:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp4_QH_QH_onlyQS/wout_final.nc')
    config = 'nfp4_QH_QH_onlyQS'
elif args.type == 3:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp2_QA_QA/wout_final.nc')
    config = 'nfp2_QA_QA_least_squares'
elif args.type == 4:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp4_QH_QH/wout_final.nc')
    config = 'nfp4_QH_QH_least_squares'

s_radius = 0.25
alpha_fieldline = 0
nphi= 99#141
nlambda = 37#33
nperiod = 2.0#5.0
nstep = 280
dt = 0.4
aky_min = 0.4
aky_max = 3.0
naky = 6
ngauss = 3
negrid = 8
vnewk = 0.01
phi_GS2 = np.linspace(-nperiod*np.pi, nperiod*np.pi, nphi)

# Define output directories and create them if they don't exist
OUT_DIR = os.path.join(this_path,results_folder,config,figures_directory)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

# START
vmec = Vmec(vmec_file)
fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=True, show=False)
plt.savefig(os.path.join(OUT_DIR,f'{config}_geometry_profiles_s{s_radius}_alpha{alpha_fieldline}.pdf'))
plt.close()

matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
data = [fl1.grad_alpha_dot_grad_alpha[0,0,:], fl1.grad_s_dot_grad_s[0,0,:], fl1.modB[0,0,:], fl1.B_cross_grad_B_dot_grad_alpha[0,0,:]]
parameters = [r'$|\nabla \alpha|^2$', r'$|\nabla \psi|^2$', r'$|B|$', r'\textbf B $(\times \nabla B) \cdot \nabla \alpha$']
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
    plt.savefig(os.path.join(OUT_DIR,f'{config}_{save_name}_s{s_radius}_alpha{alpha_fieldline}.pdf'))
    plt.close()