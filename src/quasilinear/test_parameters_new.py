#!/usr/bin/env python3
import os
import subprocess
import netCDF4
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from simsopt.mhd import Vmec
import argparse
from quasilinear_gs2 import quasilinear_estimate
from to_gs2 import to_gs2
import matplotlib.pyplot as plt

# Constants and Configurations
THIS_PATH = Path(__file__).parent.resolve()
GS2_EXECUTABLE = '/Users/rogeriojorge/local/gs2/bin/gs2'
CONFIG = {
    1: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/wout_nfp2_QA.nc',
        "output_dir": 'test_out_nfp2_QA_initial'
    },
    2: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/wout_nfp4_QH.nc',
        "output_dir": 'test_out_nfp4_QH_initial'
    }
}

PARAMS = {
    'nphi': 99,
    'nlambda': 37,
    'nperiod': 2.0,
    'nstep': 280,
    'dt': 0.4,
    'aky_min': 0.4,
    'aky_max': 3.0,
    'naky': 6,
    'LN': 1.0,
    'LT': 3.0,
    's_radius': 0.25,
    'alpha_fieldline': 0,
    'ngauss': 3,
    'negrid': 8,
    'vnewk': 0.01
}

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=2)
args = parser.parse_args()
config = CONFIG[args.type]
OUTPUT_DIR = THIS_PATH / f"{config['output_dir']}_ln{PARAMS['LN']}_lt{PARAMS['LT']}"
OUTPUT_CSV = OUTPUT_DIR / f"{config['output_dir']}_ln{PARAMS['LN']}_lt{PARAMS['LT']}.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)
vmec = Vmec(config['vmec_file'])


def plot_data(x, y, xlabel, ylabel, filename, label="", legend_loc=0, legend_fontsize=14, figsize=(7.5, 4.0)):
    plt.figure(figsize=figsize)
    plt.plot(x, y, '.', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc, fontsize=legend_fontsize)
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
    plt.savefig(filename)
    plt.close()

def getgamma(stellFile, fractionToConsider=0.3):
    with netCDF4.Dataset(stellFile, 'r', mmap=False) as f:
        phi2 = np.log(f.variables['phi2'][()])
        t = f.variables['t'][()]
        startIndex = int(len(t) * (1 - fractionToConsider))
        mask = np.isfinite(phi2)
        data_x = t[mask]
        data_y = phi2[mask]
        fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
        GrowthRate = fit[0] / 2
        omega_avg_array = np.array(f.variables['omega_average'][()])
        max_index = np.nanargmax(omega_avg_array[-1, :, 0, 1])
        gamma, omega = omega_avg_array[-1, max_index, 0, 1:3]
    plot_data(t, phi2, r'$t$', r'$\ln |\hat \phi|^2$', f"{stellFile}_phi2.png", label=r'data - $\gamma_{GS2} = $' + str(gamma))
    return GrowthRate, abs(omega)

def eigenPlot(stellFile):
    with netCDF4.Dataset(stellFile, 'r', mmap=False) as f:
        y = f.variables['phi'][()]
        x = f.variables['theta'][()]
        omega_avg_array = np.array(f.variables['omega_average'][()])
        max_index = np.nanargmax(omega_avg_array[-1, :, 0, 1])
        phiR0, phiI0 = y[max_index, 0, int((len(x) - 1) / 2 + 1), :2]
        phi02 = phiR0 ** 2 + phiI0 ** 2
        phiR = (y[max_index, 0, :, 0] * phiR0 + y[max_index, 0, :, 1] * phiI0) / phi02
        phiI = (y[max_index, 0, :, 1] * phiR0 - y[max_index, 0, :, 0] * phiI0) / phi02
    plot_data(x, phiR, r'$\theta$', r'$\hat \phi$', f"{stellFile}_eigenphi.png", label=r'Re($\hat \phi/\hat \phi_0$)')
    plot_data(x, phiI, r'$\theta$', r'$\hat \phi$', f"{stellFile}_eigenphi.png", label=r'Im($\hat \phi/\hat \phi_0$)')
    return 0

def gammabyky(stellFile, fractionToConsider=0.3):
    with netCDF4.Dataset(stellFile, 'r', mmap=False) as fX:
        tX = fX.variables['t'][()]
        kyX = fX.variables['ky'][()]
        phi2_by_kyX = fX.variables['phi2_by_ky'][()]
        omegaX = fX.variables['omega'][()]
        startIndexX = int(len(tX) * (1 - fractionToConsider))
        growthRateX = [np.polyfit(tX[mask], np.log(phi2[mask]), 1)[0] / 2 for mask, phi2 in zip(np.isfinite(phi2_by_kyX).T, phi2_by_kyX.T)]
        realFrequencyX = [np.mean(omega[startIndexX:]) for omega in omegaX[:, :, 0, 0]]
    plot_data(kyX, growthRateX, r'$k_y$', r'$\gamma$', f"{stellFile}_GammaOmegaKy.png")
    plot_data(kyX, realFrequencyX, r'$k_y$', r'$\omega$', f"{stellFile}_GammaOmegaKy.png")
    return np.array(kyX), np.array(growthRateX), np.array(realFrequencyX)

def replace(file_path, pattern, subst):
    with open(file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace(pattern, subst)
    with open(file_path, 'w') as file:
        file.write(filedata)

def create_gs2_inputs(p):
    nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, aky_min, aky_max, naky, vnewk = p
    gridout_file = OUTPUT_DIR / f'grid_gs2_nphi{nphi}_nperiod{nperiod}.out'
    phi_GS2 = np.linspace(-nperiod * np.pi, nperiod * np.pi, nphi)
    to_gs2(gridout_file, vmec, PARAMS['s_radius'], PARAMS['alpha_fieldline'], phi1d=phi_GS2, nlambda=nlambda)
    gs2_input_name = f"gs2Input_nphi{nphi}_nperiod{nperiod}_nlambda{nlambda}negrid{negrid}ngauss{ngauss}_nstep{nstep}_dt{dt}_kymin{aky_min}_kymax{aky_max}_nky{naky}_ln{PARAMS['LN']}_lt{PARAMS['LT']}"
    gs2_input_file = OUTPUT_DIR / f"{gs2_input_name}.in"
    shutil.copy(THIS_PATH / '..' / 'GK_inputs' / 'gs2Input-linear.in', gs2_input_file)
    replace_dict = {
        ' gridout_file = "grid.out"': f' gridout_file = "{gridout_file}"',
        ' nstep = 150 ! Maximum number of timesteps': f' nstep = {nstep} ! Maximum number of timesteps"',
        ' fprim = 1.0 ! -1/n (dn/drho)': f' fprim = {PARAMS["LN"]} ! -1/n (dn/drho)',
        ' tprim = 3.0 ! -1/T (dT/drho)': f' tprim = {PARAMS["LT"]} ! -1/T (dT/drho)',
        ' delt = 0.4 ! Time step': f' delt = {dt} ! Time step',
        ' aky_min = 0.4': f' aky_min = {aky_min}',
        ' aky_max = 5.0': f' aky_max = {aky_max}',
        ' naky = 4': f' naky = {naky}',
        ' vnewk = 0.01 ! collisionality parameter': f' vnewk = {vnewk} ! collisionality parameter',
        ' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.': f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.',
        ' negrid = 10 ! Total number of energy grid points': f' negrid = {negrid} ! Total number of energy grid points'
    }
    for k, v in replace_dict.items():
        replace(gs2_input_file, k, v)
    return gs2_input_name

def output_to_csv(data):
    df = pd.DataFrame(data=[data])
    if not OUTPUT_CSV.exists():
        df.to_csv(OUTPUT_CSV, index=False)
    else:
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

def run_gs2(p):
    nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, aky_min, aky_max, naky, vnewk = p
    gs2_input_name = create_gs2_inputs(p)
    subprocess.run([GS2_EXECUTABLE, f"{gs2_input_name}.in"], stdout=subprocess.DEVNULL)
    output_file = OUTPUT_DIR / f"{gs2_input_name}.out.nc"
    eigenPlot(output_file)
    growth_rate, omega = getgamma(output_file)
    _, growthRateX, _ = gammabyky(output_file)
    weighted_growth_rate = np.sum(quasilinear_estimate(output_file, show=True, savefig=True)) / naky
    data = {
        'ln': PARAMS['LN'],
        'lt': PARAMS['LT'],
        'nphi': nphi,
        'nperiod': nperiod,
        'nlambda': nlambda,
        'nstep': nstep,
        'dt': dt,
        'growth_rate': growth_rate,
        'weighted_growth_rate': weighted_growth_rate,
        'negrid': negrid,
        'ngauss': ngauss,
        'aky_min': aky_min,
        'aky_max': aky_max,
        'naky': naky,
        'vnewk': vnewk
    }
    output_to_csv(data)
    for ext in ['amoments', 'eigenfunc', 'error', 'fields', 'g', 'lpc', 'mom2', 'moments', 'vres', 'vres2', 'exit_reason', 'optim', 'out', 'in', 'gs2Input', 'out.nc']:
        for f in OUTPUT_DIR.glob(f"*.{ext}"):
            f.unlink()
    return growth_rate, weighted_growth_rate

def main():
    param_list = [
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (2*PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], 2*PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], 2*PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], 2*PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], 2*PARAMS['nstep'], 0.5*PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], 2*PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], 2*PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], 0.5*PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], 2*PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], 2*PARAMS['naky'], PARAMS['vnewk']),
        (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], 0.5*PARAMS['vnewk']),
    ]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_gs2, param_list))

    for i, (growth_rate, sum_gamma) in enumerate(results):
        print(f'Run {i + 1}: growth_rate={growth_rate:.1f}, sum(gamma/ky)={sum_gamma:.1f}')

if __name__ == "__main__":
    main()
