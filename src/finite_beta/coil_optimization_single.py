#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from simsopt import save, load
from scipy.optimize import minimize
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux, Weight
from simsopt.field import BiotSavart, Current, coils_via_symmetries, coils_to_focus, coils_to_makegrid
from simsopt.geo import (CurveLength, curves_to_vtk, create_equally_spaced_curves,
                         SurfaceRZFourier, CurveLength, CurveCurveDistance, ArclengthVariation,
                         MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber)
this_path = os.path.dirname(os.path.abspath(__file__))

QA_or_QH = "QH"
beta = 2.5
filename = 'wout_final.nc'
results_folder = 'results'

ncoils = 6
R0 = 11.3
R1 = 4.3
order = 12
MAXITER = 1000

start_from_sratch = False # True if starting from circular coils, False if starting from a previous optimization

MSC_THRESHOLD = 1.
if start_from_sratch:
    max_length_per_coil = 30
    CC_THRESHOLD = 0.5
    CS_THRESHOLD = 0.6
    CURVATURE_THRESHOLD = 1.5
    ALS_THRESHOLD = 0.1
else:
    # copy bs.json file from coils folder to this_path
    max_length_per_coil = 41
    CC_THRESHOLD = 0.90
    CS_THRESHOLD = 1.4
    CURVATURE_THRESHOLD = 1.1
    ALS_THRESHOLD = 0.3

LENGTH_WEIGHT = 1.
CC_WEIGHT = 1.
CS_WEIGHT = 1.
CURVATURE_WEIGHT = 1.
MSC_WEIGHT = 1.
ALS_WEIGHT = 1.

output_intermediate_curves = False
output_interval = 250

nphi = 32
ntheta = 32
vc_src_nphi = 80

prefix_save = 'optimization'
OUT_DIR_APPENDIX=f"{prefix_save}_{QA_or_QH}_beta{beta:.1f}"
OUT_DIR = os.path.join(this_path,results_folder,QA_or_QH,OUT_DIR_APPENDIX)
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)
vmec_file = os.path.join(OUT_DIR,filename)

# Directory for output
out_dir = Path("coils")
out_dir.mkdir(parents=True, exist_ok=True)
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
print('virtual casing data file:', vc_filename)
if os.path.isfile(vc_filename):
    print('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
s = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
s_full = SurfaceRZFourier.from_wout(vmec_file, range="full torus", nphi=nphi*2*s.nfp, ntheta=ntheta)

if start_from_sratch:
    total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)
    base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
    base_currents = [Current(total_current / ncoils * 1e-6) * 1e6 for _ in range(ncoils-1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
else:
    bs_initial_file = f"biot_savart_nfp{s.nfp}_{QA_or_QH}_ncoils{ncoils}_order{order}.json"
    print('Loading initial Biot-Savart object:', bs_initial_file)
    if os.path.isfile(bs_initial_file):
        bs_temporary = load(bs_initial_file)
        base_curves = [bs_temporary.coils[i]._curve for i in range(ncoils)]
        base_currents = [bs_temporary.coils[i]._current for i in range(ncoils)]
    else:
        # output error and exit
        print(f"Error: {bs_initial_file} does not exist")
        exit()
    base_curves = [bs_temporary.coils[i]._curve for i in range(ncoils)]
    base_currents = [bs_temporary.coils[i]._current for i in range(ncoils)]
print('base_currents:', [current.get_value()/1e6 for current in base_currents], 'MA')
# Create the initial coils
# base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, out_dir / "curves_init")
curves_to_vtk(base_curves, out_dir / "curves_init_half_nfp")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(out_dir / "surf_init_half_nfp", extra_data=pointData)
s_full.to_vtk(out_dir / "surf_init")

Jf = SquaredFlux(s, bs, target=vc.B_external_normal, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
linkNum = LinkingNumber(curves)
Jals = [ArclengthVariation(c) for c in base_curves]

J_LENGTH = LENGTH_WEIGHT * sum(QuadraticPenalty(J, max_length_per_coil, "max") for J in Jls)
J_CC = CC_WEIGHT * Jccdist
J_CS = CS_WEIGHT * Jcsdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
J_ALS = ALS_WEIGHT * sum(QuadraticPenalty(J, ALS_THRESHOLD, "max") for J in Jals)

JF = Jf \
    + J_LENGTH \
    + J_CC \
    + J_ALS \
    + J_CURVATURE \
    + J_CS \
    # + J_MSC \
    # + linkNum

def fun(dofs, info={'Nfeval':0}):
    info['Nfeval'] += 1
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, max(B·n)/B={BdotN_max:.1e}"#, ⟨B·n⟩/B={BdotN_mean:.1e}"
    cl_string = ",".join([f"{J.J():.1f}" for J in Jls])
    outstr += f", L=[{cl_string}], "
    outstr += f'lCC={float(Jccdist.shortest_distance()):.2f}, '
    outstr += f"lCS={Jcsdist.shortest_distance():.2f}, "
    # outstr += f", ║∇J║={np.linalg.norm(grad):.1e}, "
    kap_string = ",".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    outstr +=f"k=[{kap_string}], "
    # msc_string = ",".join(f"{j.J():.1f}" for j in Jmscs)
    # outstr +=f"msc=[{msc_string}], "
    outstr +=f"J_L={J_LENGTH.J():.1e}, "
    outstr +=f"J_CC={J_CC.J():.1e}, "
    outstr +=f"J_K={J_CURVATURE.J():.1e}, "
    # outstr +=f"J_MSC={J_MSC.J():.1e}, "
    outstr +=f"J_ALS={J_ALS.J():.1e}, "
    outstr +=f"J_CS={J_CS.J():.1e}, "
    # outstr +=f"Link Number = {linkNum.J()}, "
    print(outstr)

    if np.mod(info['Nfeval'],output_interval)==0 and output_intermediate_curves==True:
        curves_to_vtk(curves, out_dir / f"curves_intermediate_{info['Nfeval']}")
        curves_to_vtk(base_curves, out_dir / f"curves_intermediate_{info['Nfeval']}_half_nfp")
        s.to_vtk(out_dir / f"surf_intermediate_{info['Nfeval']}_half_nfp", extra_data={"B_N": BdotN[:, :, None]})
        s_full.x = s.x
        s_full.to_vtk(out_dir / f"surf_intermediate_{info['Nfeval']}")

    return J, grad

dofs = JF.x
res = minimize(fun, dofs, args=({'Nfeval':0},), jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20)
dofs = res.x
curves_to_vtk(curves, out_dir / "curves_opt")
curves_to_vtk(base_curves, out_dir / "curves_opt_half_nfp")
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(out_dir / "surf_opt_half_nfp", extra_data=pointData)
s_full.x = s.x
s_full.to_vtk(out_dir / "surf_opt")

save(bs, out_dir / f"biot_savart_nfp{s.nfp}_{QA_or_QH}_ncoils{ncoils}_order{order}.json")
coils_to_makegrid(os.path.join(out_dir,"coils_makegrid_format.txt"),base_curves,base_currents,nfp=s.nfp, stellsym=True)
coils_to_focus(os.path.join(out_dir,"coils_focus_format.txt"),curves=[coil._curve for coil in coils],currents=[coil._current for coil in coils],nfp=s.nfp,stellsym=True)