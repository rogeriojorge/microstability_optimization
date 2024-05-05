#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import netcdf_file
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
args = parser.parse_args()
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
this_path = Path(__file__).parent.resolve()

wfQ_array = [100, 10, 1, 0.1, 0]
legend_fontsize = 8
phi_plot = 0

prefix_save = 'optimization'
results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
optimizer = 'least_squares'

s_radius = 0.25
alpha_fieldline = 0
phi_GS2 = np.linspace(-PARAMS['nperiod']*np.pi, PARAMS['nperiod']*np.pi, PARAMS['nphi'])

OUT_DIR_all = os.path.join(this_path,results_folder,config['output_dir'])
W7X_directory = os.path.join(this_path,results_folder,'W7-X')
figures_directory = os.path.join(OUT_DIR_all, f'figures')
os.makedirs(figures_directory, exist_ok=True)

neo_epseff_array = []
neo_sradial_array = []
simple_time_array = []
simple_confpart_pass_array = []
simple_confpart_trap_array = []
LT_array = []
weighted_growth_rate_array_at_ln_0 = []
growth_rate_array_at_ln_0 = []
ky_array_at_ln_0 = []
Aminor_array = []
alpha_array = []
weighted_growth_rate_array_at_s_025 = []
growth_rate_array_at_s_025 = []
ky_array_at_s_025 = []
R_array = []
Z_array = []
iota_array = []
s_array = []
grads2_array = []
phi_array = []
print("Loading parameters")
for weight_optTurbulence in wfQ_array:
    OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}_wFQ{weight_optTurbulence:.1f}"
    output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
    OUT_DIR = os.path.join(OUT_DIR_all,OUT_DIR_APPENDIX)
    os.chdir(OUT_DIR)
    
    Aminor_p = netcdf_file('wout_final.nc','r',mmap=False).variables['Aminor_p'][()]
    
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
    
    df = pd.read_csv(os.path.join(OUT_DIR,f'scan_ln_lt_{OUT_DIR_APPENDIX}.csv'))
    LT_array.append(sorted(df['lt'].unique()))
    # print(sorted(df['ln'].unique()))
    weighted_growth_rate_array_at_ln_0.append(df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['weighted_growth_rate'].first().values)
    growth_rate_array_at_ln_0.append(df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['growth_rate'].first().values)
    ky_array_at_ln_0.append(df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['ky'].first().values)
    
    df = pd.read_csv(os.path.join(OUT_DIR,f'scan_s_alpha_{OUT_DIR_APPENDIX}.csv'))
    alpha_array.append(sorted(df['alpha'].unique()))
    weighted_growth_rate_array_at_s_025.append(df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['weighted_growth_rate'].first().values)
    growth_rate_array_at_s_025.append(df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['growth_rate'].first().values)
    ky_array_at_s_025.append(df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['ky'].first().values)
    
    f = netcdf_file('wout_final.nc','r',mmap=False)
    rmnc = f.variables['rmnc'][()]
    zmns = f.variables['zmns'][()]
    xn = f.variables['xn'][()]
    xm = f.variables['xm'][()]
    ns = f.variables['ns'][()]
    iotaf = f.variables['iotaf'][()]
    s = np.linspace(0,1,ns)
    iota_array.append(iotaf)
    s_array.append(s)
    nmodes = len(xn)
    ntheta = 200
    theta = np.linspace(0,2*np.pi,num=ntheta)
    iradius = ns-1
    R = np.zeros((ntheta,))
    Z = np.zeros((ntheta,))
    for itheta in range(ntheta):
        for imode in range(nmodes):
            angle = xm[imode]*theta[itheta] - xn[imode]*phi_plot
            R[itheta] = R[itheta] + rmnc[iradius,imode]*np.cos(angle)
            Z[itheta] = Z[itheta] + zmns[iradius,imode]*np.sin(angle)
    R_array.append(R)
    Z_array.append(Z)
    
    vmec = Vmec(os.path.join(OUT_DIR, 'wout_final.nc'),verbose=False)
    fl1 = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi_GS2, plot=False, show=False)
    grads2_array.append(fl1.grad_s_dot_grad_s[0,0,:])
    phi_array.append(fl1.phi[0,0,:])

print("Saving figures")

## Epsilon Effective Plot
print("  Epsilon Effective Plot")
os.chdir(W7X_directory)
Aminor_p_W7X = netcdf_file('wout_W7-X_standard_configuration.nc','r',mmap=False).variables['Aminor_p'][()]
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
fig = plt.figure(figsize=(5, 3), dpi=200)
ax = fig.add_subplot(111)
plt.plot(s_radial_W7X,eps_eff_W7X, '--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(neo_sradial_array[i],neo_epseff_array[i], label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
ax.set_yscale('log')
plt.xlabel(r'$s=\psi/\psi_b$', fontsize=12)
plt.ylabel(r'$\epsilon_{eff}$', fontsize=14)
plt.xlim([0,1])
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
fig.savefig(os.path.join(figures_directory,f'neo_out_{config["output_dir"]}.pdf'), dpi=fig.dpi)#, bbox_inches = 'tight', pad_inches = 0)

## Fast particle confinement plot
print("  Fast particle confinement plot")
os.chdir(W7X_directory)
W7X_simple_time_array = np.loadtxt(f'time_array.txt')
W7X_simple_confpart_pass_array = np.loadtxt(f'confpart_pass_array.txt')
W7X_simple_confpart_trap_array = np.loadtxt(f'confpart_trap_array.txt')

os.chdir(figures_directory)
fig = plt.figure(figsize=(4, 3), dpi=200)
plt.semilogx(W7X_simple_time_array, 1 - (W7X_simple_confpart_pass_array + W7X_simple_confpart_trap_array), '--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.semilogx(simple_time_array[i], 1 - (simple_confpart_pass_array[i] + simple_confpart_trap_array[i]), label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlim([1e-6, 1e-2])
plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'simple_out_{config["output_dir"]}.pdf', dpi=200)

## weighted growth rate plot of LT
print("  weighted growth rate plot of LT")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_ln_lt_W7-X.csv'))
W7X_LT_array = df['lt'].unique()
W7X_weighted_growth_rate_array_at_ln_0 = df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['weighted_growth_rate'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_LT_array,W7X_weighted_growth_rate_array_at_ln_0, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(LT_array[i],weighted_growth_rate_array_at_ln_0[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$a/L_T$')
plt.ylabel(r'$\gamma/\langle k_{\perp}^2 \rangle$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'weighted_growth_rate_LT_{config["output_dir"]}.pdf', dpi=200)

## growth rate plot of LT
print("  growth rate plot of LT")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_ln_lt_W7-X.csv'))
W7X_LT_array = df['lt'].unique()
W7X_growth_rate_array_at_ln_0 = df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['growth_rate'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_LT_array,W7X_growth_rate_array_at_ln_0, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(LT_array[i],growth_rate_array_at_ln_0[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$a/L_T$')
plt.ylabel(r'$\gamma_{\textrm{max}}$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'max_growth_rate_LT_{config["output_dir"]}.pdf', dpi=200)

## ky plot of LT
print("  ky plot of LT")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_ln_lt_W7-X.csv'))
W7X_LT_array = df['lt'].unique()
W7X_ky_array_at_ln_0 = df[abs(df['ln'] - 1.0)<0.29].groupby('lt')['ky'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_LT_array,W7X_ky_array_at_ln_0, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(LT_array[i],ky_array_at_ln_0[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$a/L_T$')
plt.ylabel(r'$k_y$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'max_ky_LT_{config["output_dir"]}.pdf', dpi=200)

## weighted growth rate plot of alpha
print("  weighted growth rate plot of alpha")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_s_alpha_W7-X.csv'))
W7X_alpha_array = sorted(df['alpha'].unique())
W7X_weighted_growth_rate_array_at_s_025 = df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['weighted_growth_rate'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_alpha_array,W7X_weighted_growth_rate_array_at_s_025, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(alpha_array[i],weighted_growth_rate_array_at_s_025[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\gamma/\langle k_{\perp}^2 \rangle$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'weighted_growth_rate_alpha_{config["output_dir"]}.pdf', dpi=200)

## growth rate plot of alpha
print("  growth rate plot of alpha")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_s_alpha_W7-X.csv'))
W7X_alpha_array = sorted(df['alpha'].unique())
W7X_growth_rate_array_at_s_025 = df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['growth_rate'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_alpha_array,W7X_growth_rate_array_at_s_025, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(alpha_array[i],growth_rate_array_at_s_025[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\gamma_{\textrm{max}}$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'max_growth_rate_alpha_{config["output_dir"]}.pdf', dpi=200)

## ky plot of alpha
print("  ky plot of alpha")
os.chdir(W7X_directory)
df = pd.read_csv(os.path.join(W7X_directory,f'scan_s_alpha_W7-X.csv'))
W7X_alpha_array = sorted(df['alpha'].unique())
W7X_ky_array_at_s_025 = df[abs(df['s'] - 0.25)<0.04].groupby('alpha')['ky'].first().values

os.chdir(figures_directory)
fig = plt.figure(figsize=(4,3), dpi=200)
plt.plot(W7X_alpha_array,W7X_ky_array_at_s_025, '*--', label='W7-X')
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(alpha_array[i],ky_array_at_s_025[i], '+-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$k_y$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'max_ky_alpha_{config["output_dir"]}.pdf', dpi=200)

## Cross Section plot
print("  Cross Section plot")
fig = plt.figure(figsize=(3,4), dpi=200)
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(R_array[i], Z_array[i], '-', label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel('R')
plt.ylabel('Z')
plt.axis('equal')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'cross_section_{config["output_dir"]}.pdf', dpi=200)

## Iota plot
print("  Iota plot")
fig = plt.figure(figsize=(4,3), dpi=200)
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(s_array[i], np.abs(iota_array[i]), label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel(r'$s = \psi/\psi_b$')
plt.ylabel(r'$\iota$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'iota_{config["output_dir"]}.pdf', dpi=200)

## Geometry grads2 plot
print("  Geometry grads2 plot")
fig = plt.figure(figsize=(4,3), dpi=200)
for i, weight_optTurbulence in enumerate(wfQ_array):
    plt.plot(phi_array[i]/np.abs(iota_array[i][2])**0.85, grads2_array[i], label=r'$w_{FQ}=$'+f'{weight_optTurbulence:.1f}')
plt.xlabel('Standard toroidal angle $\phi$/Rotational transform $\iota$')
plt.ylabel(r'$|\nabla \psi|^2$')
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(f'gradpsi2_{config["output_dir"]}.pdf', dpi=200)
