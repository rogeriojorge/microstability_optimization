#!/usr/bin/env python3
import os
import sys
import shutil
import netCDF4
import argparse
import subprocess
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from quasilinear_gs2 import quasilinear_estimate
from concurrent.futures import ProcessPoolExecutor

# Constants and Configurations
THIS_PATH = Path(__file__).parent.resolve()
sys.path.insert(1, os.path.join(THIS_PATH, '..', 'util'))
from to_gs2 import to_gs2 # pylint: disable=import-error
GS2_EXECUTABLE = '/Users/rogeriojorge/local/gs2/bin/gs2'
CONFIG = {
    -3: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/wout_nfp1_QI.nc',
        "output_dir": 'nfp1_QI_initial',
        "params": { 'nphi': 69,'nlambda': 21,'nperiod': 2.0,'nstep': 220,'dt': 0.5,
                    'aky_min': 0.3,'aky_max': 4.0,'naky': 8,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    },
    -2: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/wout_nfp4_QH.nc',
        "output_dir": 'nfp4_QH_initial',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    },
    -1: {
        "vmec_file": '/Users/rogeriojorge/local/microstability_optimization/src/vmec_inputs/wout_nfp2_QA.nc',
        "output_dir": 'nfp2_QA_initial',
        "params": { 'nphi': 89,'nlambda': 25,'nperiod': 3.0,'nstep': 270,'dt': 0.4,
                    'aky_min': 0.4,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
    }
}
prefix_save = 'test_convergence'
results_folder = 'results'
n_processors_default = 4

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=-2)
parser.add_argument("--nprocessors", type=int, default=n_processors_default, help="Number of processors to use for parallel execution")
args = parser.parse_args()
config = CONFIG[args.type]
PARAMS = config['params']
OUTPUT_DIR = os.path.join(THIS_PATH,results_folder,config['output_dir'],f"{prefix_save}_{config['output_dir']}")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{prefix_save}_{config['output_dir']}.csv")
# OUTPUT_DIR = THIS_PATH / f"{config['output_dir']}_ln{PARAMS['LN']}_lt{PARAMS['LT']}"
# OUTPUT_CSV = OUTPUT_DIR / f"{config['output_dir']}_ln{PARAMS['LN']}_lt{PARAMS['LT']}.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OUTPUT_DIR)
vmec = Vmec(config['vmec_file'])

def getgamma(stellFile, fractionToConsider=0.3):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    phi2 = np.log(f.variables['phi2'][()])
    t = f.variables['t'][()]
    startIndex = int(len(t)*(1-fractionToConsider))
    mask = np.isfinite(phi2)
    data_x = t[mask]
    data_y = phi2[mask]
    fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
    poly = np.poly1d(fit)
    GrowthRate = fit[0]/2
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_omega = omega_average_array[-1,:,0,0]
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    gamma = omega_average_array_gamma[max_index]
    omega = omega_average_array_omega[max_index]
    # gamma  = np.mean(f.variables['omega'][()][startIndex:,0,0,1])
    # omega  = np.mean(f.variables['omega'][()][startIndex:,0,0,0])
    #fitRes = np.poly1d(coeffs)
    # if not os.path.exists(stellFile+'_phi2.pdf'):
    plt.figure(figsize=(7.5,4.0))
    ##############
    plt.plot(t, phi2,'.', label=r'data - $\gamma_{GS2} = $'+str(gamma))
    plt.plot(t, poly(t),'-', label=r'fit - $\gamma = $'+str(GrowthRate))
    ##############
    plt.legend(loc=0,fontsize=14)
    plt.xlabel(r'$t$');plt.ylabel(r'$\ln |\hat \phi|^2$')
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.97)
    plt.savefig(stellFile+'_phi2.png')
    plt.close()
    return GrowthRate, abs(omega)
# Save final eigenfunction
def eigenPlot(stellFile):
    f = netCDF4.Dataset(stellFile,'r',mmap=False)
    y = f.variables['phi'][()]
    x = f.variables['theta'][()]
    plt.figure(figsize=(7.5,4.0))
    omega_average_array = np.array(f.variables['omega_average'][()])
    omega_average_array_gamma = omega_average_array[-1,:,0,1]
    max_index = np.nanargmax(omega_average_array_gamma)
    phiR0= y[max_index,0,int((len(x)-1)/2+1),0]
    phiI0= y[max_index,0,int((len(x)-1)/2+1),1]
    phi02= phiR0**2+phiI0**2
    phiR = (y[max_index,0,:,0]*phiR0+y[max_index,0,:,1]*phiI0)/phi02
    phiI = (y[max_index,0,:,1]*phiR0-y[max_index,0,:,0]*phiI0)/phi02
    ##############
    plt.plot(x, phiR, label=r'Re($\hat \phi/\hat \phi_0$)')
    plt.plot(x, phiI, label=r'Im($\hat \phi/\hat \phi_0$)')
    ##############
    plt.xlabel(r'$\theta$');plt.ylabel(r'$\hat \phi$')
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.16, bottom=0.19, right=0.98, top=0.93)
    plt.savefig(stellFile+'_eigenphi.png')
    plt.close()
    return 0
##### Function to obtain gamma and omega for each ky
def gammabyky(stellFile,fractionToConsider=0.3):
    # Compute growth rate:
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['t'][()]
    kyX  = fX.variables['ky'][()]
    phi2_by_kyX  = fX.variables['phi2_by_ky'][()]
    omegaX  = fX.variables['omega'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    growthRateX  = []
    ## assume that kyX=kyNA
    for i in range(len(kyX)):
        maskX  = np.isfinite(phi2_by_kyX[:,i])
        data_xX = tX[maskX]
        data_yX = phi2_by_kyX[maskX,i]
        fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
        thisGrowthRateX  = fitX[0]/2
        growthRateX.append(thisGrowthRateX)
    # Compute real frequency:
    realFreqVsTimeX  = []
    realFrequencyX   = []
    for i in range(len(kyX)):
        realFreqVsTimeX.append(omegaX[:,i,0,0])
        realFrequencyX.append(np.mean(realFreqVsTimeX[i][startIndexX:]))
    numRows = 1
    numCols = 2

    plt.subplot(numRows, numCols, 1)
    plt.plot(kyX,growthRateX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\gamma$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size='xx-small'),loc=0)

    plt.subplot(numRows, numCols, 2)
    plt.plot(kyX,realFrequencyX,'.-')
    plt.xlabel(r'$k_y$')
    plt.ylabel(r'$\omega$')
    plt.xscale('log')
    plt.rc('font', size=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    # plt.legend(frameon=False,prop=dict(size=12),loc=0)

    plt.tight_layout()
    #plt.subplots_adjust(left=0.14, bottom=0.15, right=0.98, top=0.96)
    plt.savefig(stellFile+"_GammaOmegaKy.png")
    plt.close()
    return np.array(kyX), np.array(growthRateX), np.array(realFrequencyX)

def replace(file_path, pattern, subst):
    with open(file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace(pattern, subst)
    with open(file_path, 'w') as file:
        file.write(filedata)

def create_gs2_inputs(p):
    nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, aky_min, aky_max, naky, vnewk = p
    nphi=int(nphi); nlambda=int(nlambda); nstep=int(nstep); negrid=int(negrid); ngauss=int(ngauss); naky=int(naky)
    gridout_file = str(os.path.join(OUTPUT_DIR,f"grid_gs2_nphi{nphi}_nperiod{nperiod}_nlambda{nlambda}negrid{negrid}ngauss{ngauss}vnewk{vnewk}_nstep{nstep}_dt{dt}_kymin{aky_min}_kymax{aky_max}_nky{naky}_ln{PARAMS['LN']}_lt{PARAMS['LT']}.out"))
    phi_GS2 = np.linspace(-nperiod * np.pi, nperiod * np.pi, nphi)
    to_gs2(gridout_file, vmec, PARAMS['s_radius'], PARAMS['alpha_fieldline'], phi1d=phi_GS2, nlambda=nlambda)
    gs2_input_name = f"gs2Input_nphi{nphi}_nperiod{nperiod}_nlambda{nlambda}negrid{negrid}ngauss{ngauss}vnewk{vnewk}_nstep{nstep}_dt{dt}_kymin{aky_min}_kymax{aky_max}_nky{naky}_ln{PARAMS['LN']}_lt{PARAMS['LT']}"
    gs2_input_file = str(os.path.join(OUTPUT_DIR,f"{gs2_input_name}.in"))
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
    if not Path(OUTPUT_CSV).exists():
        df.to_csv(OUTPUT_CSV, index=False)
    else:
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

def run_gs2(p):
    start_time = time()
    nphi, nperiod, nlambda, nstep, dt, negrid, ngauss, aky_min, aky_max, naky, vnewk = p
    nphi=int(nphi); nlambda=int(nlambda); nstep=int(nstep); negrid=int(negrid); ngauss=int(ngauss); naky=int(naky)
    gs2_input_name = create_gs2_inputs(p)
    # subprocess.run([GS2_EXECUTABLE, f"{gs2_input_name}.in"], stdout=subprocess.DEVNULL)
    proc = subprocess.Popen(f"{GS2_EXECUTABLE} {gs2_input_name}.in".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
    proc.wait()
    output_file = os.path.join(OUTPUT_DIR,f"{gs2_input_name}.out.nc")
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
    print(f'nphi={nphi} nperiod={nperiod} nlambda={nlambda} nstep={nstep} dt={dt} negrid={negrid} ngauss={ngauss} aky_min={aky_min} aky_max={aky_max} naky={naky} vnewk={vnewk} growth_rate={growth_rate:1f} sum(gamma/ky)={weighted_growth_rate:1f} took {(time()-start_time):1f}s')
    return growth_rate, weighted_growth_rate

def main():
    param_list = [
        # (PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
        (2*PARAMS['nphi']-1, PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']),
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

    ## RUN BASE CASE
    run_gs2((((PARAMS['nphi'], PARAMS['nperiod'], PARAMS['nlambda'], PARAMS['nstep'], PARAMS['dt'], PARAMS['negrid'], PARAMS['ngauss'], PARAMS['aky_min'], PARAMS['aky_max'], PARAMS['naky'], PARAMS['vnewk']))))

    n_processors = min(args.nprocessors, len(param_list))
    with ProcessPoolExecutor(max_workers=n_processors) as executor:
        results = list(executor.map(run_gs2, param_list))

    for ext in ['amoments', 'eigenfunc', 'error', 'fields', 'g', 'lpc', 'mom2', 'moments', 'vres', 'vres2', 'exit_reason', 'optim', 'out', 'in', 'vspace_integration_error', 'gs2Input', 'out.nc']:
        for f in Path(OUTPUT_DIR).glob(f"*.{ext}"):
            f.unlink()
        for f in Path(OUTPUT_DIR).glob(f".*"):
            f.unlink()

if __name__ == "__main__":
    main()
