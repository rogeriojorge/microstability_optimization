#!/usr/bin/env python3
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import matplotlib
import warnings
import matplotlib.cbook
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=0)
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()

if args.type == 0:
    vmec_file = os.path.join(this_path, '..', 'vmec_inputs', 'wout_nfp4_QH.nc')
    name = 'geometry_nfp4_QH_initial'
elif args.type == 1:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp2_QA_QA_onlyQS/wout_final.nc')
    name = 'geometry_nfp2_QA_QA_onlyQS'
elif args.type == 2:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp4_QH_QH_onlyQS/wout_final.nc')
    name = 'geometry_nfp4_QH_QH_onlyQS'
elif args.type == 3:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp2_QA_QA/wout_final.nc')
    name = 'geometry_nfp2_QA_QA_least_squares'
elif args.type == 4:
    vmec_file = os.path.join(this_path, 'output_MAXITER350_least_squares_nfp4_QH_QH/wout_final.nc')
    name = 'geometry_nfp4_QH_QH_least_squares'


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
this_path = Path(__file__).parent.resolve()
figures_directory = 'figures'
out_dir = os.path.join(this_path, figures_directory)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# START
vmec = Vmec(vmec_file)
fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=True, show=False)
plt.savefig(os.path.join(out_dir,f'{name}_geometry_profiles_s{s_radius}_alpha{alpha_fieldline}.png'));plt.close()