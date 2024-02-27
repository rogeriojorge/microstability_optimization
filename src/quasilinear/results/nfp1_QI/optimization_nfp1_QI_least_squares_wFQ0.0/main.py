#!/usr/bin/env python3
import os
import sys
import glob
import time
import shutil
import netCDF4
import argparse
import subprocess
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from tempfile import mkstemp
from functools import partial
from datetime import datetime
from os import fdopen, remove
import matplotlib.pyplot as plt
from shutil import move, copymode
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from quasilinear_gs2 import quasilinear_estimate
from scipy.optimize import dual_annealing
from configurations import CONFIG
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(this_path, '..', 'util'))
home_directory = os.path.expanduser("~")
from to_gs2 import to_gs2 # pylint: disable=import-error
import vmecPlot2 # pylint: disable=import-error
import booz_plot # pylint: disable=import-error
from qi_functions import QuasiIsodynamicResidual, MaxElongationPen, MirrorRatioPen  # pylint: disable=import-error
mpi = MpiPartition()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=5)
parser.add_argument("--wfQ", type=float, default=10)
args = parser.parse_args()
start_time = time.time()
#########
## FOR GS2 to work with SIMSOPT's MPI, USE_MPI should be undefined during GS2's compilation
## override USE_MPI = 
#########
#########
## To run this file with 4 cores, use the following command:
## mpirun -n 4 python3 main.py --type 1
## where type 1 is QA nfp2, type 2 is QH nfp4, type 3 is QI nfp1, type 4 is QA nfp3, type 5 is QH nfp3, type 6 is QI nfp2, type 7 is QI nfp3
############################################################################
#### Input Parameters
############################################################################
gs2_executable = f'{home_directory}/local/gs2/bin/gs2'
# gs2_executable = '/marconi/home/userexternal/rjorge00/gs2/bin/gs2'
MAXITER =10#150
max_modes = [3]
maxmodes_mpol_mapping = {1: 5, 2: 5, 3: 5, 4: 6, 5: 7}
prefix_save = 'optimization'
results_folder = 'results'
config = CONFIG[args.type]
PARAMS = config['params']
opt_quasisymmetry = True if (config['output_dir'][-2:] == 'QA' or config['output_dir'][-2:] == 'QH') else False
weighted_growth_rate = True #use sum(gamma/ky) instead of peak(gamma)

s_radius = 0.25
alpha_fieldline = 0
phi_GS2 = np.linspace(-PARAMS['nperiod']*np.pi, PARAMS['nperiod']*np.pi, PARAMS['nphi'])

plot_result = True
use_previous_results_if_available = False

weight_mirror = 10
iota_QA = 0.42
weight_iota_QA = 5e0
iota_QH=-0.8
weight_iota_QH=1e-4
iota_QI=0.615
weight_iota_QI=1e-4
elongation_weight = 1e2
mirror_weight = 1e2
weight_optTurbulence = args.wfQ#30
optimizer = 'least_squares'
rel_step_factor_1 = 3e-2#1e-1
max_rel_step_factor_2 = 3e-3
#diff_rel_step = 1e-1 ## diff_rel_step = 0.1/max_mode
#diff_abs_step = 1e-2 ## diff_abs_step = (max_mode/2)*10**(-max_mode)
MAXITER_LOCAL = 3
MAXFUN_LOCAL = 30
ftol=1e-6
no_local_search = False
HEATFLUX_THRESHOLD = 1e18
GROWTHRATE_THRESHOLD = 10
aspect_ratio_weight = 3e+0
diff_method = 'forward'
local_optimization_method = 'lm' # 'trf'
perform_extra_solve = True
######################################
######################################
OUT_DIR_APPENDIX=f"{prefix_save}_{config['output_dir']}_{optimizer}"
OUT_DIR_APPENDIX+=f'_wFQ{weight_optTurbulence:.1f}'
output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
OUT_DIR = os.path.join(this_path,results_folder,config['output_dir'],OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copyfile(os.path.join(this_path,'main.py'),os.path.join(OUT_DIR,'main.py'))
######################################
dest = os.path.join(OUT_DIR,OUT_DIR_APPENDIX+'_previous')
if use_previous_results_if_available and (os.path.isfile(os.path.join(OUT_DIR,'input.final')) or os.path.isfile(os.path.join(dest,'input.final'))):
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(dest, exist_ok=True)
        if os.path.isfile(os.path.join(OUT_DIR, 'input.final')) and not os.path.isfile(os.path.join(dest, 'input.final')):
            files = os.listdir(OUT_DIR)
            for f in files:
                shutil.move(os.path.join(OUT_DIR, f), dest)
    else:
        time.sleep(0.2)
    filename = os.path.join(dest, 'input.final')
else:
    filename = config['input_file']
os.chdir(OUT_DIR)
vmec = Vmec(filename, verbose=False, mpi=mpi)
vmec.keep_all_files = True
surf = vmec.boundary
######################################
def output_dofs_to_csv(dofs,mean_iota,aspect,growth_rate,quasisymmetry_total,mirror_ratio):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry_total'],['mirror_ratio']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry_total],[mirror_ratio]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
######################################
##### QI FUNCTIONS #####
######################################
snorms = [1/16, 5/16, 9/16, 13/16] # Flux surfaces at which the penalty will be calculated
nphi_QI=141 # Number of points along measured along each well
nalpha_QI=27 # Number of wells measured
nBj_QI=51 # Number of bounce points measured
mpol_QI=18 # Poloidal modes in Boozer transformation
ntor_QI=18 # Toroidal modes in Boozer transformation
nphi_out_QI=2000 # size of return array if arr_out_QI = True
arr_out_QI=True # If True, returns (nphi_out*nalpha) values, each of which is the difference
maximum_elongation = 6 # Defines the maximum elongation allowed in the QI elongation objective function
maximum_mirror = 0.19 # Defines the maximum mirror ratio of |B| allowed in the QI elongation objective function
optQI = partial(QuasiIsodynamicResidual,snorms=snorms, nphi=nphi_QI, nalpha=nalpha_QI, nBj=nBj_QI, mpol=mpol_QI, ntor=ntor_QI, nphi_out=nphi_out_QI, arr_out=arr_out_QI)
qi = make_optimizable(optQI, vmec)
partial_MaxElongationPen = partial(MaxElongationPen,t=maximum_elongation)
optElongation = make_optimizable(partial_MaxElongationPen, vmec)
partial_MirrorRatioPen = partial(MirrorRatioPen,t=maximum_mirror)
optMirror = make_optimizable(partial_MirrorRatioPen, vmec)
########################################
########################################
##### CALCULATE GROWTH RATE HERE #######
########################################
########################################
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
def CalculateGrowthRate(v: Vmec):
    try:
        v.run()
        f_wout = v.output_file.split('/')[-1]
        gs2_input_name = f"gs2-{f_wout[5:-3]}"
        gs2_input_file = os.path.join(OUT_DIR,f'{gs2_input_name}.in')
        shutil.copy(os.path.join(this_path,'..','GK_inputs','gs2Input-linear.in'),gs2_input_file)
        gridout_file = os.path.join(OUT_DIR,f'grid_{gs2_input_name}.out')
        replace(gs2_input_file,' gridout_file = "grid.out"',f' gridout_file = "grid_{gs2_input_name}.out"')
        replace(gs2_input_file,' nstep = 150',f' nstep = {PARAMS["nstep"]}')
        replace(gs2_input_file,' delt = 0.4 ! Time step',f' delt = {PARAMS["dt"]} ! Time step')
        replace(gs2_input_file,' fprim = 1.0 ! -1/n (dn/drho)',f' fprim = {PARAMS["LN"]} ! -1/n (dn/drho)')
        replace(gs2_input_file,' tprim = 3.0 ! -1/T (dT/drho)',f' tprim = {PARAMS["LT"]} ! -1/T (dT/drho)')
        replace(gs2_input_file,' aky_min = 0.4',f' aky_min = {PARAMS["aky_min"]}')
        replace(gs2_input_file,' aky_max = 5.0',f' aky_max = {PARAMS["aky_max"]}')
        replace(gs2_input_file,' naky = 4',f' naky = {PARAMS["naky"]}')
        replace(gs2_input_file,' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
        f' ngauss = {PARAMS["ngauss"]} ! Number of untrapped pitch-angles moving in one direction along field line.')
        replace(gs2_input_file,' negrid = 10 ! Total number of energy grid points',
        f' negrid = {PARAMS["negrid"]} ! Total number of energy grid points')
        to_gs2(gridout_file, v, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=PARAMS["nlambda"])
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        # f_log = os.path.join(OUT_DIR,f"{gs2_input_name}.log")
        # with open(f_log, 'w') as fp:
        p = subprocess.Popen(bashCommand.split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)#stdout=fp)
        p.wait()
        # subprocess.call(bashCommand, shell=True)
        fractionToConsider = 0.3 # fraction of time from the simulation period to consider
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
        weighted_growth_rate = np.sum(quasilinear_estimate(os.path.join(OUT_DIR,f"{gs2_input_name}.out.nc")))/PARAMS["naky"]

        if not np.isfinite(qavg): qavg = HEATFLUX_THRESHOLD
        if not np.isfinite(growth_rate): growth_rate = HEATFLUX_THRESHOLD
        if not np.isfinite(weighted_growth_rate): weighted_growth_rate = HEATFLUX_THRESHOLD

    except Exception as e:
        pprint(e)
        qavg = HEATFLUX_THRESHOLD
        growth_rate = GROWTHRATE_THRESHOLD
        weighted_growth_rate = GROWTHRATE_THRESHOLD

    try: os.remove(os.path.join(OUT_DIR,f'{v.input_file.split("/")[-1]}_{gs2_input_name[-10:]}'))
    except Exception as e: pass
    try: os.remove(os.path.join(OUT_DIR,v.output_file))
    except Exception as e: pass
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
######################################
######################################
######################################
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
    if "QA" in config["output_dir"]:
        qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    else:
        qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    quasisymmetry_total = qs.total()
    if np.isnan(quasisymmetry_total) or quasisymmetry_total>1e18: return GROWTHRATE_THRESHOLD
    str = f'{datetime.now().strftime("%H:%M:%S")} - '
    if weighted_growth_rate:
        str += f'sum(gamma/ky) = {growth_rate:.2f}, '
    else:
        str += f'peak(gamma) = {growth_rate:.2f}, '
    if opt_quasisymmetry:
        str += f'quasisymmetry = {quasisymmetry_total:.3f}, '
        mirror_ratio = 0
    else:
        mirror_ratio = MirrorRatioPen(v)
        str += f'mirror = {mirror_ratio:1f}, '
    str +=  f'with aspect ratio={v.aspect():.2f} and iota={v.mean_iota():.2f} took {(time.time()-start_time):.3f}s'
    print(str)
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),growth_rate,quasisymmetry_total,mirror_ratio)
    return growth_rate
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
######################################
try:
    pprint("Initial aspect ratio:", vmec.aspect())
    pprint("Initial mean iota:", vmec.mean_iota())
    pprint("Initial magnetic well:", vmec.vacuum_well())
except Exception as e: pprint(e)
#if MPI.COMM_WORLD.rank == 0: pprint("Initial growth rate:", CalculateGrowthRate(vmec))
######################################
initial_dofs=np.copy(surf.x)
def fun(dofss):
    prob.x = dofss
    objective = prob.objective()

    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"input*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"gs2-*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f".gs2-*")): os.remove(objective_file)
    except Exception as e: pass
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR,f"grid_gs2-*")): os.remove(objective_file)
    except Exception as e: pass

    return objective
for max_mode in max_modes:
    if (not opt_quasisymmetry) and (max_mode==1):
        continue
    output_path_parameters=f'output_{optimizer}_maxmode{max_mode}.csv'
    vmec.indata.mpol = maxmodes_mpol_mapping[max_mode]
    vmec.indata.ntor = maxmodes_mpol_mapping[max_mode]
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################  
    opt_tuple = [(vmec.aspect, config["aspect_ratio_target"], aspect_ratio_weight)]
    if weight_optTurbulence>0.01: opt_tuple.append((optTurbulence.J, 0, weight_optTurbulence))
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    if "QA" in config["output_dir"]:
        qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
        opt_tuple.append((vmec.mean_iota, iota_QA, weight_iota_QA))
    elif "QH" in config["output_dir"]:
        opt_tuple.append((vmec.mean_iota, iota_QH, weight_iota_QH))
    if opt_quasisymmetry:
        opt_tuple.append((qs.residuals, 0, 1))
    else:
        opt_tuple.append((vmec.mean_iota, iota_QI, weight_iota_QI))
        opt_tuple.append((qi.J, 0, 1))
        opt_tuple.append((optElongation.J, 0, elongation_weight))
        opt_tuple.append((optMirror.J, 0, mirror_weight))
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    #pprint('## Now calculating total objective function ##')
    #if MPI.COMM_WORLD.rank == 0: pprint("Total objective before optimization:", prob.objective())
    pprint('-------------------------')
    pprint(f'Optimizing with max_mode = {max_mode}')
    pprint('-------------------------')
    if optimizer == 'dual_annealing':
        initial_temp = 1000
        visit = 2.0
        bounds = [(-0.25,0.25) for _ in dofs]
        minimizer_kwargs = {"method": "Nelder-Mead", "bounds": bounds, "options": {'maxiter': MAXITER_LOCAL, 'maxfev': MAXFUN_LOCAL, 'disp': True}}
        if MPI.COMM_WORLD.rank == 0: res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, initial_temp=initial_temp,visit=visit, no_local_search=no_local_search, x0=dofs, minimizer_kwargs=minimizer_kwargs)
    elif optimizer == 'least_squares':
        diff_rel_step = rel_step_factor_1/max_mode
        diff_abs_step = min(max_rel_step_factor_2,(max_mode/5)*10**(-max_mode))
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER, ftol=ftol)#, diff_method=diff_method, method=local_optimization_method)
        if perform_extra_solve: least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step/10, abs_step=diff_abs_step/10, max_nfev=MAXITER, ftol=ftol)#, diff_method=diff_method, method=local_optimization_method)
    else: print('Optimizer not available')
    ######################################
    try: 
        pprint("Final aspect ratio:", vmec.aspect())
        pprint("Final mean iota:", vmec.mean_iota())
        pprint("Final magnetic well:", vmec.vacuum_well())
        pprint("Final quasisymmetry:", qs.total())
        pprint("Final quasi-isodynamic:", np.sum(qi.J()))
        #if MPI.COMM_WORLD.rank == 0: pprint("Final growth rate:", CalculateGrowthRate(vmec))
        if MPI.COMM_WORLD.rank == 0: vmec.write_input(os.path.join(OUT_DIR, f'input.max_mode{max_mode}'))
    except Exception as e: pprint(e)
    ######################################
if MPI.COMM_WORLD.rank == 0: vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
### PLOT RESULT
######################################
if plot_result and MPI.COMM_WORLD.rank==0:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000, 10000, 40000]#,  5000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
    vmec_final.run()
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    pprint('Creating vmec plots for vmec_final')
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    pprint('Creating Boozer plots for vmec_final')
    b1 = Boozer(vmec_final, mpol=64, ntor=64)
    boozxform_nsurfaces=10
    print('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    print(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    b1.run()
    # b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn.nc"))
    print("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
##############################################################################
##############################################################################
pprint(f'Whole optimization took {(time.time()-start_time):1f}s')
### Remove unnecessary files
patterns = ["objective_*", "residuals_*", "jac_*", "wout_nfp*", "input.nfp*", "jxbout_nfp*", "mercier.nfp*", "threed*", "parvmecinfo*"]
for pattern in patterns:
    try:
        for objective_file in glob.glob(os.path.join(OUT_DIR, pattern)):
            os.remove(objective_file)
    except Exception as e:
        pass
