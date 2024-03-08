#!/usr/bin/env python3

# This script runs coil optimizations, one after another, choosing the weights
# and target values from a random distribution. This is effectively a crude form
# of global optimization.

import os
import json
import numpy as np
from scipy.optimize import minimize
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    curves_to_vtk,
    create_equally_spaced_curves,
    SurfaceRZFourier,
    LinkingNumber,
    CurveLength,
    CurveCurveDistance,
    MeanSquaredCurvature,
    LpCurveCurvature,
    CurveSurfaceDistance,
)
from simsopt.objectives import SquaredFlux, QuadraticPenalty
this_path = os.path.dirname(os.path.abspath(__file__))

QA_or_QH = "QH"
beta = 2.5
filename = 'wout_final.nc'
results_folder = 'results'
ncoils = 3
MAXITER = 700
R1_mean = 4.2
R1_std = 0.8
min_length_per_coil = 30
max_length_per_coil = 45
min_curvature = 0.6
max_curvature = 8
CC_min = 0.6
CC_max = 1.2
order_min = 5
order_max = 15
# surface and virtual casing resolution
nphi = 32
ntheta = 32
vc_src_nphi = 80
# not using coil-surface distance
CS_THRESHOLD = 0.3
CS_WEIGHT = 10

# Directories
prefix_save = 'optimization'
OUT_DIR_APPENDIX=f"{prefix_save}_{QA_or_QH}_beta{beta:.1f}"
OUT_DIR = os.path.join(this_path,results_folder,QA_or_QH,OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
vmec_file = os.path.join(OUT_DIR,filename)

# File for the target plasma surface. It can be either a wout or vmec input file.
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
if os.path.isfile(vc_filename):
    print('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

out_dir = os.path.join(OUT_DIR,"coils","scan")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

# Load the target plasma surface:
basename = os.path.basename(vmec_file)
if basename[:4] == "wout": surf = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
else: surf = SurfaceRZFourier.from_vmec_input(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
nfp = surf.nfp
R0 = surf.get_rc(0, 0)

# Create a copy of the surface that is closed in theta and phi, and covers the
# full torus toroidally. This is nice for visualization.
nphi_big = nphi * 2 * nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi = np.linspace(0, 1, nphi_big)
surf_big = SurfaceRZFourier(
    dofs=surf.dofs,
    nfp=nfp,
    mpol=surf.mpol,
    ntor=surf.ntor,
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta,
)


def run_optimization(
    R1,
    order,
    length_target,
    length_weight,
    max_curvature_threshold,
    max_curvature_weight,
    msc_threshold,
    msc_weight,
    cc_threshold,
    cc_weight,
    index,
):
    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_length_target_{length_target:.2}_weight_{length_weight:.2}"
        + f"_max_curvature_{max_curvature_threshold:.2}_weight_{max_curvature_weight:.2}"
        + f"_msc_{msc_threshold:.2}_weight_{msc_weight:.2}"
        + f"_cc_{cc_threshold:.2}_weight_{cc_weight:.2}"
    )

    print()
    print("***********************************************")
    print(f"Job {index+1}")
    print("Parameters:", directory)
    print("***********************************************")
    print()

    # Directory for output
    new_OUT_DIR = directory + "/"
    os.mkdir(directory)

    # Create the initial coils:
    base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=order * 16)
    # # base_currents = [Current(1e5) for i in range(ncoils)]
    # base_currents = [Current(1.0) * (1e5) for i in range(ncoils)]
    # base_currents[0].fix_all()
    total_current = Vmec(vmec_file).external_current() / (2*nfp)
    base_currents = [Current(total_current / ncoils * 1e-6) * 1e6 for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]

    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    curves = [c.curve for c in coils]
    curves_to_vtk(curves, new_OUT_DIR + "curves_init", close=True)
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    pointData = {"B.N/B": BdotN[:, :, None]}
    surf.to_vtk(new_OUT_DIR + "surf_init", extra_data=pointData)

    surf_big.to_vtk(new_OUT_DIR + "surf_big")

    # Define the individual terms objective function:
    Jf = SquaredFlux(surf, bs, target=vc.B_external_normal, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, cc_threshold, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, surf, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, max_curvature_threshold) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added:
    JF = (
        Jf
        + length_weight * QuadraticPenalty(sum(Jls), length_target * ncoils)
        + cc_weight * Jccdist
        # + CS_WEIGHT * Jcsdist
        + max_curvature_weight * sum(Jcs)
        + msc_weight * sum(QuadraticPenalty(J, msc_threshold, "max") for J in Jmscs)
        + LinkingNumber(curves, 2)
    )

    iteration = 0

    def fun(dofs):
        nonlocal iteration
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN = np.max(np.abs((np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)))
        outstr = f"{iteration:4}  J={J:.1e}, Jf={jf:.1e}, max⟨B·n⟩/B={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
        # outstr += f", C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        iteration += 1
        return J, grad

    res = minimize( fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": MAXITER, "maxcor": 300}, tol=1e-15)
    JF.x = res.x
    print(res.message)
    curves_to_vtk(curves, new_OUT_DIR + "curves_opt", close=True)
    curves_to_vtk(base_curves, new_OUT_DIR + "curves_opt_halfnfp", close=True)

    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    pointData = {"B.N/B": BdotN[:, :, None]}

    surf.to_vtk(new_OUT_DIR + "surf_opt", extra_data=pointData)

    # bs_big = BiotSavart(coils)
    # bs_big.set_points(surf_big.gamma().reshape((-1, 3)))
    # pointData = {
    #     "B_N": np.sum(
    #         bs_big.B().reshape((nphi_big, ntheta_big, 3)) * surf_big.unitnormal(),
    #         axis=2,
    #     )[:, :, None]
    # }
    surf_big.to_vtk(new_OUT_DIR + "surf_big_opt")#, extra_data=pointData)

    # Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
    bs.save(new_OUT_DIR + "biot_savart.json")

    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.max(np.abs(np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2))

    results = {
        "nfp": nfp,
        "R0": R0,
        "R1": R1,
        "ncoils": ncoils,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_target": length_target,
        "length_weight": length_weight,
        "max_curvature_threshold": max_curvature_threshold,
        "max_curvature_weight": max_curvature_weight,
        "msc_threshold": msc_threshold,
        "msc_weight": msc_weight,
        "JF": float(JF.J()),
        "Jf": float(Jf.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "max_curvatures": [np.max(c.kappa()) for c in base_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "cc_threshold": cc_threshold,
        "cc_weight": cc_weight,
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "directory": directory,
        "mean_squared_curvatures": [float(J.J()) for J in Jmscs],
        "max_mean_squared_curvature": float(max(J.J() for J in Jmscs)),
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "coil_currents": [c.get_value() for c in base_currents],
        "coil_surface_distance":  float(Jcsdist.shortest_distance()),
    }

    with open(new_OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)


#########################################################################
# Carry out the scan. Below you can adjust the ranges for the random weights and
# thresholds.
#########################################################################


def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min


for index in range(10000):
    # Initial radius of the coils:
    R1 = np.random.rand() * R1_std + R1_mean

    # Number of Fourier modes describing each Cartesian component of each coil:
    order = int(np.round(rand(order_min, order_max)))

    # Target length (per coil!) and weight for the length term in the objective function:
    length_target = rand(min_length_per_coil, max_length_per_coil)
    length_weight = 10.0 ** rand(-1, 1)

    # Threshold and weight for the curvature penalty in the objective function:
    max_curvature_threshold = rand(min_curvature, max_curvature)
    max_curvature_weight = 10.0 ** rand(-6, -4)

    # Threshold and weight for the mean squared curvature penalty in the objective function:
    msc_threshold = rand(min_curvature, max_curvature)
    msc_weight = 10.0 ** rand(-7, -4)

    # Threshold and weight for the coil-to-coil distance penalty in the objective function:
    cc_threshold = rand(CC_min, CC_max)
    cc_weight = 10.0 ** rand(-1, 4)

    run_optimization(
        R1,
        order,
        length_target,
        length_weight,
        max_curvature_threshold,
        max_curvature_weight,
        msc_threshold,
        msc_weight,
        cc_threshold,
        cc_weight,
        index,
    )
