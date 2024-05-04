#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import netCDF4
import subprocess
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from multiprocessing import Pool
from quasilinear_gs2 import quasilinear_estimate
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
import matplotlib
import warnings
import matplotlib.cbook
import argparse
from configurations import CONFIG
#########
## To run this file with 8 cores, use the following command:
## python3 check2_scan_s_alpha.py --type 2 --wfQ 10
## where type 1 is QA nfp2, type 2 is QH nfp4, type 3 is QI nfp1, type 4 is QA nfp3, type 5 is QH nfp3, type 6 is QI nfp2, type 7 is QI nfp3, type 8 is QI nfp4
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
parser.add_argument("--wfQ", type=float, default=10)
args = parser.parse_args()
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
this_path = Path(__file__).parent.resolve()
sys.path.insert(1, os.path.join(this_path, '..', 'util'))
from to_gs2 import to_gs2  # pylint: disable=import-error
######## INPUT PARAMETERS ########
home_directory = os.path.expanduser("~")
gs2_executable = f'{home_directory}/local/gs2/bin/gs2'
# gs2_executable = '/marconi/home/userexternal/rjorge00/gs2/bin/gs2'
prefix_save = 'optimization'
results_folder = 'results_March1_2024'
config = CONFIG[args.type]
PARAMS = config['params']
weighted_growth_rate = True #use sum(gamma/ky) instead of peak(gamma)
weight_optTurbulence = args.wfQ
optimizer = 'least_squares'
n_processes_parallel = 14
n_points = 8

plot_extent_fix_gamma = True
plot_gamma_max = 0.5
plot_extent_fix_weighted_gamma = True
plot_weighted_gamma_max = 0.5

s_radius_array = np.linspace(0.1, 1, n_points)
alpha_fieldline_array = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
# # Add the desired values to the arrays
desired_s_radius = 0.25
desired_alpha = 0
# s_radius_array = np.sort(np.append(s_radius_array, desired_s_radius))
# alpha_fieldline_array = np.sort(np.append(alpha_fieldline_array, desired_alpha))

phi_GS2 = np.linspace(-PARAMS['nperiod']*np.pi, PARAMS['nperiod']*np.pi, PARAMS['nphi'])
########################################
# Go into the output directory
if config['output_dir']=='W7-X':
    OUT_DIR_APPENDIX=config['output_dir']
    OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'])
    vmec = Vmec(os.path.join(OUT_DIR, 'wout_W7-X_standard_configuration.nc'),verbose=False)
else:
    OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}"
    OUT_DIR_APPENDIX+=f'_wFQ{weight_optTurbulence:.1f}'
    OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],OUT_DIR_APPENDIX)
    vmec = Vmec(os.path.join(OUT_DIR, 'wout_final.nc'),verbose=False)
output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
output_csv = os.path.join(OUT_DIR,f'scan_s_alpha_{OUT_DIR_APPENDIX}.csv')
#### Auxiliary functions
# Get growth rates
def getgamma(stellFile, fractionToConsider=0.3, savefig=False):
    f = netCDF4.Dataset(stellFile, 'r', mmap=False)
    phi2 = np.log(f.variables['phi2'][()])
    t = f.variables['t'][()]
    startIndex = int(len(t) * (1 - fractionToConsider))
    mask = np.isfinite(phi2)
    data_x = t[mask]
    data_y = phi2[mask]
    fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
    poly = np.poly1d(fit)
    GrowthRate = fit[0] / 2
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_omega = np.mean(omega_average_array[startIndex:,:,0,0], axis=0)
    omega_average_array_gamma = np.mean(omega_average_array[startIndex:,:,0,1], axis=0)
    max_index = np.nanargmax(omega_average_array_gamma)
    gamma = omega_average_array_gamma[max_index]
    omega = omega_average_array_omega[max_index]
    kyX  = f.variables['ky'][()]
    ky_max = kyX[max_index]

    if savefig:
        plt.figure(figsize=(7.5, 4.0))
        plt.plot(t, phi2, '.', label=r'data - $\gamma_{GS2} = $' + str(gamma))
        plt.plot(t, poly(t), '-', label=r'fit - $\gamma = $' + str(GrowthRate))
        plt.legend(loc=0, fontsize=14)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\ln |\hat \phi|^2$')
        plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
        plt.savefig(stellFile + '_phi2.png')
        plt.close()
    return GrowthRate, omega, ky_max

# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile, 'r', mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure()
    omega_average_array = np.array(f.variables['omega_average'][()])
    fractionToConsider = 0.3
    tX   = f.variables['t'][()]
    startIndexX  = int(len(tX) * (1 - fractionToConsider))
    omega_average_array_gamma = np.mean(omega_average_array[startIndexX:,:,0,1], axis=0)
    max_index = np.nanargmax(omega_average_array_gamma)
    phiR0 = y[max_index,0,int((len(x)) / 2),0]
    phiI0 = y[max_index,0,int((len(x)) / 2),1]
    phi02 = phiR0**2 + phiI0**2
    phiR = (y[max_index,0,:,0] * phiR0 + y[max_index,0,:,1] * phiI0) / phi02
    phiI = (y[max_index,0,:,1] * phiR0 - y[max_index,0,:,0] * phiI0) / phi02
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$)')
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$)')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\hat \phi$')
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile + '_eigenphi.png')
    plt.close()
    return 0

##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile, fractionToConsider=0.3, savefig=False):
    # Compute growth rate:
    fX = netCDF4.Dataset(stellFile, 'r', mmap=False)
    tX = fX.variables['t'][()]
    kyX = fX.variables['ky'][()]
    phi2_by_kyX = fX.variables['phi2_by_ky'][()]
    omegaX = fX.variables['omega'][()]
    startIndexX = int(len(tX) * (1 - fractionToConsider))
    growthRateX = []
    ## assume that kyX=kyNA
    for i in range(len(kyX)):
        maskX = np.isfinite(phi2_by_kyX[:,i])
        data_xX = tX[maskX]
        data_yX = phi2_by_kyX[maskX,i]
        fitX = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
        thisGrowthRateX = fitX[0] / 2
        growthRateX.append(thisGrowthRateX)
    # Compute real frequency:
    realFreqVsTimeX = []
    realFrequencyX = []
    for i in range(len(kyX)):
        realFreqVsTimeX.append(omegaX[:,i,0,0])
        realFrequencyX.append(np.mean(realFreqVsTimeX[i][startIndexX:]))
    if savefig:
        numRows = 1
        numCols = 2

        plt.subplot(numRows, numCols, 1)
        plt.plot(kyX, growthRateX, '.-')
        plt.xlabel(r'$k_y$')
        plt.ylabel(r'$\gamma$')
        plt.xscale('log')

        plt.subplot(numRows, numCols, 2)
        plt.plot(kyX, realFrequencyX, '.-')
        plt.xlabel(r'$k_y$')
        plt.ylabel(r'$\omega$')
        plt.xscale('log')

        plt.tight_layout()
        plt.savefig(stellFile + "_GammaOmegaKy.png")
        plt.close()
    return np.array(kyX), np.array(growthRateX), np.array(realFrequencyX)

# Function to replace text in a file
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)

def output_to_csv(growth_rate, omega, ky, weighted_growth_rate, ln, lt, s, alpha):
    keys = np.concatenate([['ln'], ['lt'], ['growth_rate'], ['omega'], ['ky'], ['weighted_growth_rate'], ['s'], ['alpha']])
    values = np.concatenate([[ln], [lt], [growth_rate], [omega], [ky], [weighted_growth_rate], [s], [alpha]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
    df.to_csv(output_csv, mode='a', header=False, index=False)

# Run GS2
growth_rate_array = np.zeros((len(s_radius_array), len(alpha_fieldline_array)))
omega_array = np.zeros((len(s_radius_array), len(alpha_fieldline_array)))
ky_array = np.zeros((len(s_radius_array), len(alpha_fieldline_array)))
weighted_growth_rate_array = np.zeros((len(s_radius_array), len(alpha_fieldline_array)))
def run_gs2(s_radius, alpha):
    start_time_local = time()
    try:
        grid_file_name = f'grid_gs2_s{s_radius:.3f}-alpha{alpha:.3f}.out'
        gridout_file = os.path.join(OUT_DIR, grid_file_name)
        to_gs2(gridout_file, vmec, s_radius, alpha, phi1d=phi_GS2, nlambda=PARAMS["nlambda"])
        fl1 = vmec_fieldlines(vmec, s_radius, alpha, phi1d=phi_GS2, plot=False, show=False)
        # plt.savefig(f'geometry_profiles_s{s_radius}_alpha{alpha}.png')
        # plt.close()
        gs2_input_name = f"gs2Input-s{s_radius:.3f}-alpha{alpha:.3f}"
        gs2_input_file = os.path.join(OUT_DIR, f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path, '..', 'GK_inputs', 'gs2Input-linear.in'), gs2_input_file)
        replace(gs2_input_file, ' gridout_file = "grid.out"', f' gridout_file = "{grid_file_name}"')
        replace(gs2_input_file,' nstep = 150',f' nstep = {PARAMS["nstep"]}')
        replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {PARAMS["dt"]} ! Time step')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {PARAMS["LN"]} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {PARAMS["LT"]} ! -1/T (dT/drho)')
        replace(gs2_input_file,' aky_min = 0.4',f' aky_min = {PARAMS["aky_min"]}')
        replace(gs2_input_file,' aky_max = 5.0',f' aky_max = {PARAMS["aky_max"]}')
        replace(gs2_input_file,' naky = 4',f' naky = {PARAMS["naky"]}')
        replace(gs2_input_file,' vnewk = 0.01 ! collisionality parameter',f' vnewk = {PARAMS["vnewk"]} ! collisionality parameter')
        replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
            f' ngauss = {PARAMS["ngauss"]} ! Number of untrapped pitch-angles moving in one direction along field line.')
        replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
            f' negrid = {PARAMS["negrid"]} ! Total number of energy grid points')
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        p = subprocess.Popen(bashCommand.split(), stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
        p.wait()
        file2read = os.path.join(OUT_DIR, f"{gs2_input_name}.out.nc")
        if abs(s_radius-desired_s_radius)<0.05 and abs(alpha-desired_alpha)<0.1:
            show_fig=True
            save_fig=True
            eigenPlot(file2read)
        else:
            show_fig=False
            save_fig=False
        growth_rate, omega, ky = getgamma(file2read,savefig=save_fig)
        kyX, growthRateX, realFrequencyX = gammabyky(file2read,savefig=save_fig)
        weighted_growth_rate = np.sum(quasilinear_estimate(file2read,show=show_fig,savefig=save_fig))/PARAMS["naky"]
        output_to_csv(growth_rate, omega, ky, weighted_growth_rate, PARAMS["LN"], PARAMS["LT"], s_radius, alpha)
    except Exception as e:
        print(e)
        exit()
    print(f'  alpha={alpha:1f}, s_radius={s_radius:1f}, growth rate={growth_rate:1f}, omega={omega:1f}, ky={ky:1f}, weighted gamma={weighted_growth_rate:1f} took {(time()-start_time_local):1f}s')
    return growth_rate, omega, ky, weighted_growth_rate

def parallel_run_gs2(params):
    s, alpha = params
    return run_gs2(s, alpha)

if __name__ == '__main__':
    print('Starting GS2 scan')
    start_time = time()
    pool = Pool(processes=n_processes_parallel)
    params_list = [(s, alpha) for s in s_radius_array for alpha in alpha_fieldline_array]
    results = pool.map(parallel_run_gs2, params_list)
    pool.close()
    pool.join()

    # Unpack the results
    for i, (growth_rate, omega, ky, weighted_growth_rate) in enumerate(results):
        s_index = i // len(alpha_fieldline_array)
        alpha_index = i % len(alpha_fieldline_array)
        growth_rate_array[s_index, alpha_index] = growth_rate
        omega_array[s_index, alpha_index] = omega
        ky_array[s_index, alpha_index] = ky
        weighted_growth_rate_array[s_index, alpha_index] = weighted_growth_rate

    print(f'Running GS2 scan took {time()-start_time}s')
    
    for f in glob.glob('*.amoments'): remove(f)
    for f in glob.glob('*.eigenfunc'): remove(f)
    for f in glob.glob('*.error'): remove(f)
    for f in glob.glob('*.fields'): remove(f)
    for f in glob.glob('*.g'): remove(f)
    for f in glob.glob('*.lpc'): remove(f)
    for f in glob.glob('*.mom2'): remove(f)
    for f in glob.glob('*.moments'): remove(f)
    for f in glob.glob('*.vres'): remove(f)
    for f in glob.glob('*.vres2'): remove(f)
    for f in glob.glob('*.exit_reason'): remove(f)
    for f in glob.glob('*.optim'): remove(f)
    for f in glob.glob('*.out'): remove(f)
    for f in glob.glob('*.scratch'): remove(f)
    for f in glob.glob('*.used_inputs.in'): remove(f)
    for f in glob.glob('*.vspace_integration_error'): remove(f)
    ## THIS SHOULD ONLY REMOVE FILES STARTING WTH .gs2
    for f in glob.glob('.gs2*'): remove(f)
    ## REMOVE ALSO INPUT FILES
    for f in glob.glob('*.in'): remove(f)
    ## REMOVE ALSO OUTPUT FILES
    for f in glob.glob('*.out.nc'):
        if f not in f"gs2Input-s{desired_s_radius:.3f}-alpha{desired_alpha:.3f}.out.nc": remove(f)