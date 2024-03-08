#!/usr/bin/env python3
import os
import sys
import time
import glob
import shutil
import argparse
import numpy as np
import booz_xform as bx
import matplotlib.pyplot as plt
from simsopt.mhd.vmec import Vmec
from simsopt import make_optimizable
from simsopt.mhd.boozer import Boozer
from simsopt.util.mpi import MpiPartition, log
from simsopt.solve import least_squares_mpi_solve
from simsopt.objectives import LeastSquaresProblem
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.mhd.profiles import ProfilePolynomial, ProfilePressure, ProfileScaled, ProfileSpline
from simsopt.util.constants import ELEMENTARY_CHARGE
from simsopt.mhd.bootstrap import RedlGeomBoozer, VmecRedlBootstrapMismatch, j_dot_B_Redl, RedlGeomVmec
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(this_path, '..', 'util'))
import vmecPlot2 # pylint: disable=import-error
from qi_functions import MaxElongationPen
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
args = parser.parse_args()

QA_or_QH = 'QH' if args.type == 1 else 'QA'
MAXITER = 50
optimize_well = True
optimize_DMerc = True
optimize_shear = True
optimize_elongation = False
plot_result = True
max_modes = [2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]

beta = 2.5 #%
diff_method = 'centered'
abs_step = 2.0e-5
rel_step = 2.0e-3
maxmodes_mpol_mapping = {1: 3, 2: 5, 3: 6, 4: 6, 5: 6}
ftol = 1e-6
xtol = 1e-7
aspect_ratio = 6.5
shear_min_QA = 0.10
shear_min_QH = 0.12
shear_weight = 1e-2
iota_min_QA = 0.42
iota_min_QH = 1.05
iota_Weight = 1e2
well_Weight = 1e5
DMerc_Weight = 1e17
elongation_target = 6
elongation_weight = 1e-2
# Should we target a max/min elongation?
opt_method = 'trf'#'lm'
DMerc_fraction = 0.40 # The starting radius of the Mercier criterion minimum find (0<...<1)

# Set up Vmec object
home_directory = os.path.expanduser("~")
vmec_folder = f'{home_directory}/local/microstability_optimization/src/vmec_inputs'
filename = os.path.join(vmec_folder,"input.nfp4_QH_finitebeta") if QA_or_QH == 'QH' else os.path.join(vmec_folder,"input.nfp2_QA_finitebeta")
mpi = MpiPartition()

ne0 = 3e20 * (beta/100/0.05)**(1/3)
Te0 = 15e3 * (beta/100/0.05)**(2/3)
# ne0 = 4.13e20
# Te0 = 12.0e3
ne = ProfilePolynomial(ne0 * np.array([1, 0, 0, 0, 0, -1.0]))
Te = ProfilePolynomial(Te0 * np.array([1, -1.0]))

Zeff = 1.0
ni = ne
Ti = Te
pressure = ProfilePressure(ne, Te, ni, Ti)
pressure_Pa = ProfileScaled(pressure, ELEMENTARY_CHARGE)

######################################
######################################
prefix_save = 'optimization'
results_folder = 'results'
OUT_DIR_APPENDIX=f"{prefix_save}_{QA_or_QH}_beta{beta:.1f}"
OUT_DIR = os.path.join(this_path,results_folder,QA_or_QH,OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
shutil.copyfile(os.path.join(this_path,'main.py'),os.path.join(OUT_DIR,'main.py'))
os.chdir(OUT_DIR)
######################################
vmec = Vmec(filename, mpi=mpi, verbose=False)
# vmec.verbose = mpi.proc0_world
surf = vmec.boundary
vmec.unfix('phiedge')
#vmec.unfix('curtor')

vmec.pressure_profile = pressure_Pa
vmec.n_pressure = 7
vmec.indata.pcurr_type = 'cubic_spline_ip'
vmec.n_current = 50

# Define objectives for minor radius and <B>
def minor_radius_objective(vmec): return vmec.wout.Aminor_p
def volavgB_objective(vmec): return vmec.wout.volavgB
def DMerc_min_objective(vmec): return np.min((np.min(vmec.wout.DMerc[int(len(vmec.wout.DMerc) * DMerc_fraction):]),0))
def magnetic_well_objective(vmec): return np.min((vmec.vacuum_well(),0))
def iota_min_objective(vmec): return np.min((np.min(np.abs(vmec.wout.iotaf))-(iota_min_QA if QA_or_QH=='QA' else iota_min_QH),0))
def shear_objective(vmec): return np.min((np.abs(vmec.mean_shear())-(shear_min_QA if QA_or_QH=='QA' else shear_min_QH),0))
def elongation_objective(vmec): return MaxElongationPen(vmec,t=6.0,ntheta=16,nphi=6,print_all=True)

minor_radius_optimizable =  make_optimizable(minor_radius_objective, vmec)
volavgB_optimizable      =  make_optimizable(volavgB_objective, vmec)
DMerc_optimizable        =  make_optimizable(DMerc_min_objective, vmec)
magnetic_well_optimizable = make_optimizable(magnetic_well_objective, vmec)
iota_min_optimizable    =   make_optimizable(iota_min_objective, vmec)
shear_optimizable       =   make_optimizable(shear_objective, vmec)
elongation_optimizable  =   make_optimizable(elongation_objective, vmec)

# Define quasisymmetry objective:
helicity_n = -1 if QA_or_QH == 'QH' else 0
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=helicity_n)

#redl_geom_hires = RedlGeomVmec(vmec)

# Target values:
aminor_ARIES = 1.70442622782386
volavgB_ARIES = 5.86461221551616

# Define bootstrap objective:
#redl_s = np.linspace(0, 1, 22)
#redl_geom = RedlGeomVmec(vmec, redl_s[1:-1])  # Drop s=0 and s=1 to avoid problems with epsilon=0 and p=0
#prob = LeastSquaresProblem.from_tuples([(bootstrap_mismatch.residuals, 0, 1)])

if mpi.proc0_world:
    try:
        print("Initial aspect ratio:", vmec.aspect())
        print("Initial min iota:", np.min(np.abs(vmec.wout.iotaf)))
        print("Initial mean iota:", vmec.mean_iota())
        print("Initial mean shear:", vmec.mean_shear())
        print("Initial magnetic well:", vmec.vacuum_well())
        print("Initial quasisymmetry:", qs.total())
        print("Initial volavgB:", vmec.wout.volavgB)
        print("Initial min DMerc from mid radius:", DMerc_optimizable.J())
        print("Initial Aminor:", vmec.wout.Aminor_p)
        print("Initial betatotal:", vmec.wout.betatotal)
    except Exception as e: print(e)

# Fourier modes of the boundary with m <= max_mode and |n| <= max_mode
# will be varied in the optimization. A larger range of modes are
# included in the VMEC and booz_xform calculations.
max_mode_old = max_modes[0]
step_max_mode_factor = -1
for step, max_mode in enumerate(max_modes):
    if mpi.proc0_world: print(f'Optimizing with max_mode={max_mode}')
    if max_mode == max_mode_old: step_max_mode_factor+=1
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, 
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    #surf.fix("rc(0,0)") # Major radius

    n_spline = np.min((step * 2 + 5, 15))
    s_spline = np.linspace(0, 1, n_spline)
    if step == 0: current = ProfileSpline(s_spline, s_spline * (1 - s_spline) * 4)
    else: current = current.resample(s_spline)
    current.unfix_all()
    vmec.current_profile = ProfileScaled(current, -1e6)

    # Divide MPI_COMM_WORLD into worker groups
    ndofs = len(vmec.x)
    if diff_method == 'forward': ngroups = 1 + ndofs
    elif diff_method == 'centered': ngroups = 2 * ndofs
    else: raise RuntimeError('Should not get here')

    ngroups = min(ngroups, mpi.comm_world.size)
    mpi = MpiPartition(ngroups)
    # mpi.write()
    # print(flush=True)
    vmec.update_mpi(mpi)

    vmec.indata.mpol = maxmodes_mpol_mapping[max_mode]
    vmec.indata.ntor = maxmodes_mpol_mapping[max_mode]
    
    # Define bootstrap objective:
    booz = Boozer(vmec, mpol=12, ntor=12)
    ns = 50
    s_full = np.linspace(0, 1, ns)
    ds = s_full[1] - s_full[0]
    s_half = s_full[1:] - 0.5 * ds
    # Pick out the points in s_half that are closest to the points in s_spline:
    s_redl = []
    for s in s_spline:
        index = np.argmin(np.abs(s_half - s))
        s_redl.append(s_half[index])
    # if mpi.proc0_world:
    #     print(f's_spline: {s_spline}.  s_redl: {s_redl}', flush=True)
    # Make sure all points are unique:
    assert len(s_redl) == len(set(s_redl))
    assert len(s_redl) == len(s_spline)
    s_redl = np.array(s_redl)
    redl_geom = RedlGeomBoozer(booz, s_redl, helicity_n)

    # redl_s = np.linspace(0, 1, 22)
    # redl_geom = RedlGeomVmec(vmec, redl_s[1:-1])  # Drop s=0 and s=1 to avoid problems with epsilon=0 and p=0
    
    logfile = None
    if mpi.proc0_world: logfile = f'jdotB_log_max_mode{max_mode}'#_step{step}'
    bootstrap_mismatch = VmecRedlBootstrapMismatch(redl_geom, ne, Te, Ti, Zeff, helicity_n, logfile=logfile)

    opt_tuple = [(vmec.aspect, aspect_ratio, 1),
                 (minor_radius_optimizable.J, aminor_ARIES, 1),
                 (volavgB_optimizable.J, volavgB_ARIES, 1),
                 (bootstrap_mismatch.residuals, 0, 1),
                 (qs.residuals, 0, 1),
                 (iota_min_optimizable.J, 0, iota_Weight)]
    if optimize_well:  opt_tuple.append((magnetic_well_optimizable.J, 0.0, well_Weight))
    if optimize_DMerc: opt_tuple.append((DMerc_optimizable.J, 0.0, DMerc_Weight))
    if optimize_shear: opt_tuple.append((shear_optimizable.J, 0.0, shear_weight))
    if optimize_elongation: opt_tuple.append((elongation_optimizable.J, elongation_target, elongation_weight))

    prob = LeastSquaresProblem.from_tuples(opt_tuple)

    assert len(prob.x) == ndofs

    least_squares_mpi_solve(prob, mpi, grad=True, ftol=ftol, xtol=xtol, abs_step=abs_step/(10**step_max_mode_factor),
                            rel_step=rel_step/(10**step_max_mode_factor), diff_method=diff_method, max_nfev=MAXITER, method=opt_method)

    # Preserve last output file from this step:
    # vmec.files_to_delete = []

    """
    jdotB_Redl, details = j_dot_B_Redl(redl_geom_hires, ne, Te, Ti, Zeff, helicity_n)
    f = open(f'jdotB_Redl_{step}.dat', 'w')
    f.write('s, jdotB\n')
    for s, jdotB in zip(vmec.s_half_grid, jdotB_Redl):
        f.write(f'{s}, {jdotB}\n')
    f.close()
    """

    if mpi.proc0_world:
        try:
            print("Final aspect ratio:", vmec.aspect())
            print("Final min iota:", np.min(np.abs(vmec.wout.iotaf)))
            print("Final mean iota:", vmec.mean_iota())
            print("Final mean shear:", vmec.mean_shear())
            print("Final magnetic well:", vmec.vacuum_well())
            print("Final quasisymmetry:", qs.total())
            print("Final volavgB", vmec.wout.volavgB)
            print("Final min DMerc from mid radius:", DMerc_optimizable.J())
            print("Final Aminor", vmec.wout.Aminor_p)
            print("Final betatotal", vmec.wout.betatotal)
            vmec.write_input(os.path.join(OUT_DIR, f'input.max_mode{max_mode}'))
        except Exception as e: print(e)
    max_mode_old = max_mode
    ######################################
if mpi.proc0_world: vmec.write_input(os.path.join(OUT_DIR, f'input.final'))
######################################
### PLOT RESULT
######################################
if plot_result and mpi.proc0_world:
    vmec_final = Vmec(os.path.join(OUT_DIR, f'input.final'), mpi=mpi)
    vmec_final.indata.ns_array[:3]    = [  16,    51,    101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000, 10000, 40000]#,  5000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-14]#, 1e-15, 1e-15]
    vmec_final.run()
    shutil.move(os.path.join(OUT_DIR, f"wout_final_000_000000.nc"), os.path.join(OUT_DIR, f"wout_final.nc"))
    os.remove(os.path.join(OUT_DIR, f'input.final_000_000000'))
    print('Creating vmec plots for vmec_final')
    try: vmecPlot2.main(file=os.path.join(OUT_DIR, f"wout_final.nc"), name='EP_opt', figures_folder=OUT_DIR)
    except Exception as e: print(e)
    print('Creating Boozer plots for vmec_final')
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
if mpi.proc0_world:
    print(f'Whole optimization took {(time.time()-start_time):1f}s')
    ### Remove unnecessary files
    patterns = ["objective_*", "residuals_*", "jac_*", "wout_nfp*", "input.nfp*", "jxbout_nfp*", "mercier.nfp*", "threed*", "parvmecinfo*"]
    for pattern in patterns:
        try:
            for objective_file in glob.glob(os.path.join(OUT_DIR, pattern)): os.remove(objective_file)
        except Exception as e: pass
    ##############################################################################
