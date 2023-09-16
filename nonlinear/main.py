#!/usr/bin/env python3
import os
import glob
import time
import shutil
import netCDF4
import vmecPlot2
import subprocess
import numpy as np
import pandas as pd
from mpi4py import MPI
import booz_xform as bx
from pathlib import Path
from tempfile import mkstemp
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
from scipy.optimize import dual_annealing
import matplotlib
matplotlib.use('Agg') 
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
def pprint(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)
start_time = time.time()
number_of_cores = 4 # MPI.COMM_WORLD.Get_size()
############################################################################
#### Input Parameters
############################################################################
MAXITER = 150
max_modes = [3]
initial_config = 'input.nfp2_QA'# 'input.nfp2_QA' #'input.nfp4_QH'
if initial_config[-2:]=='QA': aspect_ratio_target = 6
else: aspect_ratio_target = 8
opt_quasisymmetry = True
optimizer = 'dual_annealing'#'dual_annealing' #'least_squares'
nonlinear = True
plot_result = True
use_previous_results_if_available = False
weight_optTurbulence = 100.0
diff_rel_step = 1e-4
diff_abs_step = 1e-6
MAXITER_LOCAL = 3
MAXFUN_LOCAL = 20
no_local_search = False
output_path_parameters=f'output_{optimizer}.csv'
HEATFLUX_THRESHOLD = 1e2
aspect_ratio_weight = 1e-1
gx_executable = '/m100/home/userexternal/rjorge00/gx_latest/gx'
convert_VMEC_to_GX = '/m100/home/userexternal/rjorge00/gx_latest/geometry_modules/vmec/convert_VMEC_to_GX'
##
if nonlinear:
    LN = 1.0
    LT = 3.0
    nstep = 10000
    dt = 0.2
    nzgrid = 81
    npol = 4
    desired_normalized_toroidal_flux = 0.25
    alpha_fieldline = 0
    nhermite  = 8
    nlaguerre = 4
    nu_hyper = 0.5
    D_hyper = 0.05
    ny = 50
    nx = 50
    y0 = 12.0
else:
    LN = 1.0
    LT = 3.0
    nstep = 7000
    dt = 0.03
    nzgrid = 91
    npol = 4
    desired_normalized_toroidal_flux = 0.25
    alpha_fieldline = 0
    nhermite  = 24
    nlaguerre = 12
    nu_hyper = 1.0
    D_hyper = 0.05
    ny = 40
    nx = 1
    y0 = 10
######################################
######################################
OUT_DIR_APPENDIX=f'output_MAXITER{MAXITER}_{optimizer}_{initial_config[6:]}'
if opt_quasisymmetry: OUT_DIR_APPENDIX+=f'_{initial_config[-2:]}'
if nonlinear: OUT_DIR_APPENDIX+='_nonlinear'
else: OUT_DIR_APPENDIX+='_linear'
OUT_DIR = os.path.join(this_path, OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
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
    filename = os.path.join(this_path, initial_config)
os.chdir(OUT_DIR)
vmec = Vmec(filename, verbose=False, mpi=mpi)
vmec.keep_all_files = True
surf = vmec.boundary
######################################
def output_dofs_to_csv(dofs,mean_iota,aspect,quasisymmetry_total,growth_rate,omega,ky,qflux):
    keys=np.concatenate([[f'x({i})' for i, dof in enumerate(dofs)],['mean_iota'],['aspect'],['growth_rate'],['omega'],['ky'],['quasisymmetry_total'],['qflux']])
    values=np.concatenate([dofs,[mean_iota],[aspect],[growth_rate],[omega],[ky],[quasisymmetry_total],[qflux]])
    dictionary = dict(zip(keys, values))
    df = pd.DataFrame(data=[dictionary])
    if not os.path.exists(output_path_parameters): pd.DataFrame(columns=df.columns).to_csv(output_path_parameters, index=False)
    df.to_csv(output_path_parameters, mode='a', header=False, index=False)
######################################
######################################
##### CALCULATE growth rate HERE #######
######################################
######################################
def gammabyky(stellFile, fractionToConsider=0.4):
    fX   = netCDF4.Dataset(stellFile,'r',mmap=False)
    tX   = fX.variables['time'][()]
    startIndexX  = int(len(tX)*(1-fractionToConsider))
    kyX  = fX.variables['ky'][()]
    omega_average_array = np.array(fX.groups['Special']['omega_v_time'][()])
    realFrequencyX = omega_average_array[-1,:,0,0] # only looking at one kx
    growthRateX = omega_average_array[-1,:,0,1] # only looking at one kx
    max_index = np.nanargmax(growthRateX)
    max_growthrate_omega = np.mean(omega_average_array[startIndexX:,max_index,0,0])
    max_growthrate_gamma = np.mean(omega_average_array[startIndexX:,max_index,0,1])
    max_growthrate_ky = kyX[max_index]
    return max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky
def get_qflux(stellFile, tau=100, fractionToConsider=0.4):
    fX = netCDF4.Dataset(stellFile,'r',mmap=False)
    qflux = np.nan_to_num(np.array(fX.groups['Fluxes'].variables['qflux'][:,0]))
    time = np.array(fX.variables['time'][:])
    startIndexX  = int(len(time)*(1-fractionToConsider))
    Q_avg = np.mean(qflux[startIndexX:])
    return Q_avg
# Function to replace text in a file
def replace(file_path, pattern, subst):
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    copymode(file_path, abs_path)
    remove(file_path)
    move(abs_path, file_path)
# Function to create GS2 gridout and input file
def create_gx_inputs(vmec_file):
    f_wout = vmec_file.split('/')[-1]
    # if not os.path.isfile(os.path.join(OUT_DIR,f_wout)): shutil.copy(vmec_file,os.path.join(OUT_DIR,f_wout))
    geometry_file = os.path.join(OUT_DIR,f'gx-geometry_wout_{f_wout[5:-3]}.ing')
    shutil.copy(os.path.join(this_path,'gx-geometry-sample.ing'),geometry_file)
    replace(geometry_file,'nzgrid = 32',f'nzgrid = {nzgrid}')
    replace(geometry_file,'npol = 2',f'npol = {npol}')
    replace(geometry_file,'desired_normalized_toroidal_flux = 0.12755',f'desired_normalized_toroidal_flux = {desired_normalized_toroidal_flux:.3f}')
    replace(geometry_file,'vmec_file = "wout_gx.nc"',f'vmec_file = "{f_wout}"')
    replace(geometry_file,'alpha = 0.0"',f'alpha = {alpha_fieldline}')
    if not os.path.isfile(os.path.join(OUT_DIR,'convert_VMEC_to_GX')): shutil.copy(convert_VMEC_to_GX,os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    p = subprocess.Popen(f"./convert_VMEC_to_GX gx-geometry_wout_{f_wout[5:-3]}".split(),stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL)
    p.wait()
    try: os.remove(geometry_file)
    except Exception as e: print(e)
    gridout_file = f'grid.gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux}_nt_{2*nzgrid}'
    # os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    fname = f"gxRun_wout_{f_wout[5:-3]}"
    fnamein = os.path.join(OUT_DIR,fname+'.in')
    if nonlinear: shutil.copy(os.path.join(this_path,'gx-input_nl.in'),fnamein)
    else: shutil.copy(os.path.join(this_path,'gx-input.in'),fnamein)
    replace(fnamein,' geofile = "gx_wout.nc"',f' geofile = "gx_wout_{f_wout[5:-3]}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc"')
    replace(fnamein,' gridout_file = "grid.out"',f' gridout_file = "{gridout_file}"')
    replace(fnamein,' nstep  = 9000',f' nstep  = {nstep}')
    replace(fnamein,' fprim = [ 1.0,       1.0     ]',f' fprim = [ {LN},       {LN}     ]')
    replace(fnamein,' tprim = [ 3.0,       3.0     ]',f' tprim = [ {LT},       {LT}     ]')
    replace(fnamein,' dt = 0.010',f' dt = {dt}')
    replace(fnamein,' ntheta = 80',f' ntheta = {2*nzgrid}')
    replace(fnamein,' nhermite  = 18',f' nhermite = {nhermite}')
    replace(fnamein,' nlaguerre = 10',f' nlaguerre = {nlaguerre}')
    replace(fnamein,' nu_hyper_m = 1.0',f' nu_hyper_m = {nu_hyper}')
    replace(fnamein,' nu_hyper_l = 1.0',f' nu_hyper_l = {nu_hyper}')
    replace(fnamein,' ny = 30',f' ny = {ny}')
    replace(fnamein,' nx = 1',f' nx = {nx}')
    replace(fnamein,' y0 = 20.0',f' y0 = {y0}')
    replace(fnamein,' D_hyper = 0.05',f' D_hyper = {D_hyper}')
    # if not os.path.join(OUT_DIR,f_wout)==vmec_file: os.remove(os.path.join(OUT_DIR,f_wout))
    return fname
# Function to remove spurious GX files
def remove_gx_files(gx_input_name):
    f_wout_only = gx_input_name[11:]
    # print(f'removing grid.gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}')
    try: os.remove(f'grid.gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.in')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.nc')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.log')
    except Exception as e: pass#print(e)
    try: os.remove(f'{gx_input_name}.restart.nc')
    except Exception as e: pass#print(e)
    try: os.remove(f'gx_wout_{f_wout_only}_psiN_{desired_normalized_toroidal_flux:.3f}_nt_{2*nzgrid}_geo.nc')
    except Exception as e: pass#print(e)
    # for f in glob.glob('*.restart.nc'): remove(f)
    # for f in glob.glob('*.log'): remove(f)
    # for f in glob.glob('grid.*'): remove(f)
    # for f in glob.glob('gx_wout*'): remove(f)
    # for f in glob.glob('gxRun_*'): remove(f)
    # for f in glob.glob('input.*'): remove(f)
    ## REMOVE ALSO INPUT FILE
    # for f in glob.glob('*.in'): remove(f)
    ## REMOVE ALSO OUTPUT FILE
    # for f in glob.glob(f'{gx_input_name}.nc'): remove(f)
# Function to run GS2 and extract growth rate
def run_gx(vmec: Vmec):
    gx_input_name = create_gx_inputs(vmec.output_file)
    f_log = os.path.join(OUT_DIR,gx_input_name+".log")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.mod(MPI.COMM_WORLD.Get_rank(),number_of_cores))
    print(f'On rank {MPI.COMM_WORLD.Get_rank()} and CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
    gx_cmd = [f"{gx_executable}", f"{os.path.join(OUT_DIR,gx_input_name+'.in')}", "1"]
    with open(f_log, 'w') as fp:
        p = subprocess.Popen(gx_cmd,stdout=fp)
    p.wait()
    fout = os.path.join(OUT_DIR,gx_input_name+".nc")
    try:
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky = gammabyky(fout)
        qflux = get_qflux(fout)
    except Exception as e:
        print(e)
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD
    remove_gx_files(gx_input_name)
    return max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux
######################################
######################################
######################################
def TurbulenceCostFunction(v: Vmec):
    start_time = time.time()
    try: v.run()
    except Exception as e:
        print(e)
        return HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD
    try:
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = run_gx(v)
    except Exception as e:
        print(e)
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD, HEATFLUX_THRESHOLD
    out_str = f'{datetime.now().strftime("%H:%M:%S")} - Growth rate = {max_growthrate_gamma:1f}, qflux = {qflux:1f}, quasisymmetry = {qs.total():1f} with aspect ratio={v.aspect():1f} took {(time.time()-start_time):1f}s'
    print(out_str)
    output_dofs_to_csv(v.x,v.mean_iota(),v.aspect(),qs.total(),max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux)
    if nonlinear: return qflux
    else: return max_growthrate_gamma
optTurbulence = make_optimizable(TurbulenceCostFunction, vmec)
######################################
try:
    pprint("Initial aspect ratio:", vmec.aspect())
    pprint("Initial mean iota:", vmec.mean_iota())
    pprint("Initial magnetic well:", vmec.vacuum_well())
except Exception as e: pprint(e)
if MPI.COMM_WORLD.rank == 0:
    max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = run_gx(vmec)
    if nonlinear: pprint("Initial heat flux:", qflux)
    else: pprint("Initial growth rate:", max_growthrate_gamma)
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
    try: os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
    except Exception as e: pass
    return objective
for max_mode in max_modes:
    output_path_parameters=f'output_{optimizer}_maxmode{max_mode}.csv'
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    initial_dofs=np.copy(surf.x)
    dofs=surf.x
    ######################################  
    opt_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight)]
    opt_tuple.append((optTurbulence.J, 0, weight_optTurbulence))
    if initial_config[-2:] == 'QA': qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    else: qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)    
    if opt_quasisymmetry: opt_tuple.append((qs.residuals, 0, 1))
    if initial_config[-2:]=='QA': opt_tuple.append((vmec.mean_iota, 0.42, 1))
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
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
    else: print('Optimizer not available')
    ######################################
    try: 
        pprint("Final aspect ratio:", vmec.aspect())
        pprint("Final mean iota:", vmec.mean_iota())
        pprint("Final magnetic well:", vmec.vacuum_well())
        max_growthrate_gamma, max_growthrate_omega, max_growthrate_ky, qflux = run_gx(vmec)
        if nonlinear: pprint("Final heat flux:", qflux)
        else: pprint("Final growth rate:", max_growthrate_gamma)
    except Exception as e: pprint(e)
    ######################################
# Remove final files
try: os.remove(os.path.join(OUT_DIR,'convert_VMEC_to_GX'))
except Exception as e: pprint(e)
try:
    for f in glob.glob('input.*'): remove(f)
    for f in glob.glob('wout*'): remove(f)
except Exception as e: pprint(e)
# Create final VMEC input
# if MPI.COMM_WORLD.rank == 0: 
vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
### PLOT RESULT
######################################
if plot_result and MPI.COMM_WORLD.rank==0:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000, 10000,  4000]#,  5000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
    vmec_final.run()
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    try: os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    except Exception as e: pass
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    pprint('Creating Boozer class for vmec_final')
    b1 = Boozer(vmec_final, mpol=64, ntor=64)
    boozxform_nsurfaces=10
    pprint('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    pprint(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    pprint('Running BOOZ_XFORM')
    b1.run()
    b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_single_stage.nc"))
    pprint("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = True, sqrts=True)
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
############################################################################
############################################################################
try:
    os.remove(os.path.join(OUT_DIR,"gx"))
    os.remove(os.path.join(OUT_DIR,"convert_VMEC_to_GX"))
except Exception as e:
    pprint(e)

try:
    for objective_file in glob.glob(os.path.join(OUT_DIR,f"grid.gx_wout_{initial_config[6:]}*")):
        os.remove(objective_file)
    for residuals_file in glob.glob(os.path.join(OUT_DIR,f"gx_wout_{initial_config[6:]}*")):
        os.remove(residuals_file)
    for jac_file in glob.glob(os.path.join(OUT_DIR,f"gx-{initial_config[6:]}*")):
        os.remove(jac_file)
    for threed_file in glob.glob(os.path.join(OUT_DIR,f"input.{initial_config[6:]}*")):
        os.remove(threed_file)
    for threed_file in glob.glob(os.path.join(OUT_DIR,f"wout_{initial_config[6:]}*")):
        os.remove(threed_file)
except Exception as e:
    pprint(e)
##############################################################################
##############################################################################
pprint(f'Whole optimization took {(time.time()-start_time):1f}s')
