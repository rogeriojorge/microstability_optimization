#!/usr/bin/env python3
import os
import sys
import glob
import time
import random
import shutil
import netCDF4
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import mkstemp
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from scipy.optimize import dual_annealing
from matplotlib.animation import FuncAnimation
from simsopt.mhd import Vmec
from quasilinear_gs2 import quasilinear_estimate
from simsopt import make_optimizable
THIS_PATH = Path(__file__).parent.resolve()
sys.path.insert(1, os.path.join(THIS_PATH, '..', 'util'))
from to_gs2 import to_gs2 # pylint: disable=import-error
from simsopt.mhd import QuasisymmetryRatioResidual
import argparse
this_path = Path(__file__).parent.resolve()
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=-2)
args = parser.parse_args()
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
prefix_save = 'rbc_variation'
results_folder = 'results'
n_processors_default = 4
config = CONFIG[args.type]
PARAMS = config['params']
OUTPUT_DIR = os.path.join(THIS_PATH,results_folder,config['output_dir'],f"{prefix_save}_{config['output_dir']}")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"{prefix_save}_{config['output_dir']}.csv")
weighted_growth_rate = True #use sum(gamma/ky) instead of peak(gamma)
npoints_scan = 30
min_bound = -0.15
max_bound = 0.20
run_scan = True
run_optimization = False
plot_result = True
vmec_index_scan_opt = 0
ftol = 1e-2
s_radius = 0.25
alpha_fieldline = 0
nphi= 151
nlambda = 28
nperiod = 5.0
nstep = 340
dt = 0.4
aky_min = 0.2
aky_max = 3.0
naky = 10
LN = 1.0
LT = 3.0
s_radius = 0.25
alpha_fieldline = 0
ngauss = 3
negrid = 10
phi_GS2 = np.linspace(-nperiod*np.pi, nperiod*np.pi, nphi)

HEATFLUX_THRESHOLD = 1e18
GROWTHRATE_THRESHOLD = 10

MAXITER = 10
MAXFUN = 50
MAXITER_LOCAL = 2
MAXFUN_LOCAL = 5

output_path_parameters_opt = f'opt_dofs_loss_nstep{nstep}_dt{dt}_nlambda{nlambda}.csv'
output_path_parameters_scan = f'scan_dofs_loss_nstep{nstep}_dt{dt}_nlambda{nlambda}.csv'
output_path_parameters_min = f'min_dofs_loss_nstep{nstep}_dt{dt}_nlambda{nlambda}.csv'

OUT_DIR = os.path.join(this_path,f'test_optimization_{initial_config[-7:]}_nstep{nstep}_dt{dt}_nlambda{nlambda}')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
filename = os.path.join(this_path, initial_config)
vmec = Vmec(filename, verbose=False)
if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
surf.fix("rc(0,0)")
output_to_csv = True

def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
def output_dofs_to_csv(csv_path,dofs,mean_iota,aspect,growth_rate,quasisymmetry,well,effective_1o_time=0):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry'],['well'],['effective_1o_time']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry],[well],[effective_1o_time]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(csv_path): pd.DataFrame(columns=df.columns).to_csv(csv_path, index=False)
    df.to_csv(csv_path, mode='a', header=False, index=False)
def CalculateGrowthRate(v: Vmec):
    try:
        v.run()
        f_wout = v.output_file.split('/')[-1]
        gs2_input_name = f"gs2-{f_wout[5:-3]}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'gs2Input.in'),gs2_input_file)
        gridout_file = os.path.join(OUT_DIR,f'grid_{gs2_input_name}.out')
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "grid_{gs2_input_name}.out"')
        replace(gs2_input_file,' nstep = 150',f' nstep = {nstep}')
        replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {dt} ! Time step')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {LN} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {LT} ! -1/T (dT/drho)')
        replace(gs2_input_file,' aky_min = 0.4',f' aky_min = {aky_min}')
        replace(gs2_input_file,' aky_max = 5.0',f' aky_max = {aky_max}')
        replace(gs2_input_file,' naky = 4',f' naky = {naky}')
        replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
        f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.')
        replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
        f' negrid = {negrid} ! Total number of energy grid points')
        to_gs2(gridout_file, v, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
        bashCommand = f"{GS2_EXECUTABLE} {gs2_input_file}"
        # f_log = os.path.join(OUT_DIR,f"{gs2_input_name}.log")
        # with open(f_log, 'w') as fp:
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        # subprocess.call(bashCommand, shell=True)
        fractionToConsider = 0.4 # fraction of time from the simulation period to consider
        file2read = netCDF4.Dataset(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc"),'r')
        tX = file2read.variables['t'][()]
        qparflux2_by_ky = file2read.variables['qparflux2_by_ky'][()]
        startIndexX  = int(len(tX)*(1-fractionToConsider))
        qavg = np.mean(qparflux2_by_ky[startIndexX:,0,:])
        # omega_average = file2read.variables['omega_average'][()]
        # growth_rate = np.max(np.array(omega_average)[-1,:,0,1])
        phi2 = np.log(file2read.variables['phi2'][()])
        t = file2read.variables['t'][()]
        startIndex = int(len(t)*(1-fractionToConsider))
        mask = np.isfinite(phi2)
        data_x = t[mask]
        data_y = phi2[mask]
        fit = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)
        growth_rate = fit[0]/2

        kyX  = file2read.variables['ky'][()]
        phi2_by_kyX  = file2read.variables['phi2_by_ky'][()]
        growthRateX  = []
        for i in range(len(kyX)):
            maskX  = np.isfinite(phi2_by_kyX[:,i])
            data_xX = tX[maskX]
            data_yX = phi2_by_kyX[maskX,i]
            fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
            thisGrowthRateX  = fitX[0]/2
            growthRateX.append(thisGrowthRateX)
        weighted_growth_rate = np.sum(quasilinear_estimate(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc")))/naky

        if not np.isfinite(qavg): qavg = HEATFLUX_THRESHOLD
        if not np.isfinite(growth_rate): growth_rate = HEATFLUX_THRESHOLD
        if not np.isfinite(weighted_growth_rate): weighted_growth_rate = HEATFLUX_THRESHOLD

    except Exception as e:
        print(e)
        qavg = HEATFLUX_THRESHOLD
        growth_rate = GROWTHRATE_THRESHOLD
        weighted_growth_rate = GROWTHRATE_THRESHOLD

    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"*{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f".{gs2_input_name}*")): os.remove(objective_file)
    except Exception as e: pass
    
    if weighted_growth_rate:
        return weighted_growth_rate
    else:
        return growth_rate
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return GROWTHRATE_THRESHOLD
    try:
        growth_rate = CalculateGrowthRate(v)
    except Exception as e:
        growth_rate = GROWTHRATE_THRESHOLD
    # out_str = f'{datetime.now().strftime("%H:%M:%S")} - Growth rate = {growth_rate:1f}, quasisymmetry = {qs.total():1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    out_str = f'Growth rate = {growth_rate:1f} for point {(vmec.x[vmec_index_scan_opt]):1f}, aspect {np.abs(v.aspect()):1f}, quasisymmetry = {qs.total():1f} and iota {(v.mean_iota()):1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    if output_to_csv: output_dofs_to_csv(output_path_parameters_opt, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well())
    else: output_dofs_to_csv(output_path_parameters_scan, v.x,v.mean_iota(),v.aspect(),growth_rate,qs.total(),v.vacuum_well())
    return growth_rate
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
def fun(dofss):
    vmec.x = [dofss[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
    return optTurbulence.J()
if run_optimization:
    if os.path.exists(output_path_parameters_opt): os.remove(output_path_parameters_opt)
    if os.path.exists(output_path_parameters_min): os.remove(output_path_parameters_min)
    output_to_csv = True
    bounds = [(min_bound,max_bound)]
    minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds, "options": {'maxiter': MAXITER_LOCAL, 'maxfev': MAXFUN_LOCAL, 'disp': True}}
    global_minima_found = []
    def print_fun(x, f, context):
        if context==0: context_string = 'Minimum detected in the annealing process.'
        elif context==1: context_string = 'Detection occurred in the local search process.'
        elif context==2: context_string = 'Detection done in the dual annealing process.'
        else: print(context)
        print(f'New minimum found! x={x[0]:1f}, f={f:1f}. {context_string}')
        output_dofs_to_csv(output_path_parameters_min,vmec.x,vmec.mean_iota(),vmec.aspect(),f,qs.total(),vmec.vacuum_well())
        if len(global_minima_found)>4 and np.abs((f-global_minima_found[-1])/f)<ftol:
            # Stop optimization
            return True
        else:
            global_minima_found.append(f)
    no_local_search = False
    res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, maxfun=MAXFUN, x0=[random.uniform(min_bound,max_bound)], no_local_search=no_local_search, minimizer_kwargs=minimizer_kwargs, callback=print_fun)
    print(f"Global minimum: x = {res.x}, f(x) = {res.fun}")
    vmec.x = [res.x[0] if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
if run_scan:
    if os.path.exists(output_path_parameters_scan): os.remove(output_path_parameters_scan)
    output_to_csv = False
    for point1 in np.linspace(min_bound,max_bound,npoints_scan):
        vmec.x = [point1 if count==vmec_index_scan_opt else vx for count,vx in enumerate(vmec.x)]
        growth_rate = optTurbulence.J()
if plot_result:
    df_scan = pd.read_csv(output_path_parameters_scan)

    try:
        df_opt = pd.read_csv(output_path_parameters_opt)
        fig, ax = plt.subplots()
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
        ln, = ax.plot([], [], 'ro', markersize=1)
        vl = ax.axvline(0, ls='-', color='r', lw=1)
        patches = [ln,vl]
        ax.set_xlim(min_bound,max_bound)
        ax.set_ylim(np.min(0.8*df_scan['growth_rate']), np.max(df_scan['growth_rate']))
        def update(frame):
            ind_of_frame = df_opt.index[df_opt[f'x({vmec_index_scan_opt})'] == frame][0]
            df_subset = df_opt.head(ind_of_frame+1)
            xdata = df_subset[f'x({vmec_index_scan_opt})']
            ydata = df_subset['growth_rate']
            vl.set_xdata([frame,frame])
            ln.set_data(xdata, ydata)
            return patches
        ani = FuncAnimation(fig, update, frames=df_opt[f'x({vmec_index_scan_opt})'])
        ani.save('opt_animation.gif', writer='imagemagick', fps=5)

        fig = plt.figure()
        plt.plot(df_opt[f'x({vmec_index_scan_opt})'], df_opt['growth_rate'], 'ro', markersize=1, label='Optimizer')
        plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
        plt.ylabel('Microstability Cost Function');plt.xlabel('RBC(0,1)');plt.legend();plt.savefig('growth_rate_over_opt_scan.pdf')
    except Exception as e: print(e)
    points_scan = np.linspace(min_bound,max_bound,len(df_scan[f'x({vmec_index_scan_opt})']))
    fig = plt.figure();plt.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], label='Scan')
    plt.ylabel('Microstability Cost Function');plt.xlabel('RBC(0,1)');plt.legend();plt.savefig('growth_rate_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['aspect'], label='Aspect ratio')
    plt.ylabel('Aspect ratio');plt.xlabel('RBC(0,1)');plt.savefig('aspect_ratio_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['mean_iota'], label='Rotational Transform (1/q)')
    plt.ylabel('Rotational Transform (1/q)');plt.xlabel('RBC(0,1)');plt.savefig('iota_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['quasisymmetry'], label='Quasisymmetry cost function')
    plt.ylabel('Quasisymmetry cost function');plt.xlabel('RBC(0,1)');plt.savefig('quasisymmetry_scan.pdf')
    fig = plt.figure();plt.plot(points_scan, df_scan['well'], label='Magnetic well')
    plt.ylabel('Magnetic well');plt.xlabel('RBC(0,1)');plt.savefig('magnetic_well_scan.pdf')
    # fig = plt.figure();plt.plot(points_scan, df_scan['effective_1o_time'], label='Effective 1/time')
    # plt.ylabel('Effective time');plt.xlabel('RBC(0,1)');plt.savefig('effective_1o_time_scan.pdf')

    fig=plt.figure(figsize=(8,5))
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax.set_xlabel('$RBC_{0,1}$', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    line1, = ax.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['growth_rate'], color="C0", label='$f_Q$')
    ax.set_ylabel("Microstability", color="C0", fontsize=20)
    ax.tick_params(axis='y', colors="C0", labelsize=15)
    ax.set_xlim((min_bound,max_bound))
    ax.autoscale(enable=None, axis="y", tight=False)
    line2, = ax2.plot(df_scan[f'x({vmec_index_scan_opt})'], df_scan['quasisymmetry'], color="C1", label='$f_{QS}$')
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    ax2.set_ylabel('Quasisymmetry', color="C1", fontsize=20) 
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(axis='y', colors="C1", labelsize=15)
    ax2.set_xlim((min_bound,max_bound))
    ax2.autoscale(enable=None, axis="y", tight=False)
    plt.legend(handles=[line1, line2], prop={'size': 15})
    plt.tight_layout()
    plt.savefig('quasisymmetry_vs_growthrate.pdf')
