#!/usr/bin/env python3
import os
import time
import glob
import shutil
import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import datetime
import matplotlib.pyplot as plt
from simsopt import make_optimizable
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd import QuasisymmetryRatioResidual
from inputs import (MAXITER, QA_or_QH_or_QI, nfp_QA, nfp_QH, nfp_QI,
                    aspect_ratio_QA, aspect_ratio_QH, aspect_ratio_QI,
                    opt_turbulence, results_folder, max_modes,
                    weight_aspect_ratio, weight_optTurbulence, ftol,
                    diff_method, local_optimization_method, perform_extra_solve,
                    GROWTHRATE_THRESHOLD, code_to_use)
from src.quasilinear.gs2_objective import get_gs2_objective
mpi = MpiPartition()
######################################
#### Define initial variables and ####
#### create output directory      ####
######################################
config = {'QA': (aspect_ratio_QA, nfp_QA), 'QH': (aspect_ratio_QH, nfp_QH), 'QI': (aspect_ratio_QI, nfp_QI)}
aspect_ratio_target, nfp = config.get(QA_or_QH_or_QI, (8, 4)) # Default is QH
initial_config = f'input.nfp{nfp}_{QA_or_QH_or_QI}'
OUT_DIR_APPENDIX=f'{initial_config[6:]}_MAXITER{MAXITER}'
if not opt_turbulence:  OUT_DIR_APPENDIX+=f'_onlyOmn'
this_path = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(this_path, results_folder, OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copyfile(os.path.join(this_path,'main.py'),os.path.join(OUT_DIR,'main.py'))
os.chdir(OUT_DIR)
######################################
#### Define the optimization problem #
######################################
vmec = Vmec(os.path.join(this_path, 'src', 'vmec_inputs', initial_config), verbose=False, mpi=mpi)
vmec.keep_all_files = True
surf = vmec.boundary
######## FUNCTIONS TO DEAL WITH ############
def fun(dofss):
    prob.x = dofss
    objective = prob.objective()
    try:
        file_patterns = ["input*", "wout*", "gs2-*", ".gs2-*", "grid_gs2-*"]
        for pattern in file_patterns:
            for objective_file in glob.glob(os.path.join(OUT_DIR, pattern)):
                os.remove(objective_file)
    except Exception as e:
        pass
    return objective

def output_dofs_to_csv(dofs,mean_iota,aspect,growth_rate,quasisymmetry_total):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['quasisymmetry_total'],['mirror_ratio']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[quasisymmetry_total]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)

def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return GROWTHRATE_THRESHOLD
    if code_to_use == 'gs2':
        objective = get_gs2_objective(v = v, OUT_DIR=OUT_DIR)
    else:
        print(f' Code {code_to_use} not implemented')
        raise NotImplementedError
    if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    else: qs = QuasisymmetryRatioResidual(v, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    quasisymmetry_total = qs.total()
    if np.isnan(quasisymmetry_total) or quasisymmetry_total>1e18: return GROWTHRATE_THRESHOLD
    print(f'{datetime.now().strftime("%H:%M:%S")} - {code_to_use} objective = {objective:2f}, quasisymmetry = {quasisymmetry_total:1f}, with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s')
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),objective,quasisymmetry_total)
    return objective
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
######## FUNCTIONS TO DEAL WITH ############
for max_mode in max_modes:
    output_path_parameters=f'parameters_maxmode{max_mode}.csv'
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    opt_tuple = [(vmec.aspect, aspect_ratio_target, weight_aspect_ratio)]
    if opt_turbulence: opt_tuple.append((optTurbulence.J, 0, weight_optTurbulence))
    if QA_or_QH_or_QI in ['QA','QH']:
        if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
        else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
        opt_tuple.append((qs.residuals, 0, 1))
        if initial_config[-2:] == 'QA': opt_tuple.append((vmec.mean_iota, 0.42, 1))
    else:
        raise ValueError(f'QA_or_QH_or_QI={QA_or_QH_or_QI} not implemented yet')
    prob = LeastSquaresProblem.from_tuples(opt_tuple)
    diff_rel_step = (1e-1)/max_mode
    diff_abs_step = min(1e-2,(max_mode/4)*10**(-max_mode))
    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER, ftol=ftol, diff_method=diff_method, method=local_optimization_method)
    if perform_extra_solve: least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step/10, abs_step=diff_abs_step/10, max_nfev=MAXITER, ftol=ftol, diff_method=diff_method, method=local_optimization_method)
