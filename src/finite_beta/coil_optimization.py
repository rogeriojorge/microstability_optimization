#!/usr/bin/env python3
import os
import shutil
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveLength, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier
this_path = os.path.dirname(os.path.abspath(__file__))

QA_or_QH = "QH"
beta = 2.5

filename = 'wout_final.nc'
results_folder = 'results_finally_DMerc'

ncoils = 5
R0 = 11.3
R1 = 2.0
order = 6
LENGTH_PENALTY = 1e0
MAXITER = 50

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
out_dir = Path("output")
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
total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, out_dir / "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(out_dir / "surf_init", extra_data=pointData)

Jf = SquaredFlux(s, bs, target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]

JF = Jf \
    + LENGTH_PENALTY * sum(QuadraticPenalty(Jls[i], Jls[i].J(), "identity") for i in range(len(base_curves)))

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return 1e-4*J, 1e-4*grad

dofs = JF.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20)
dofs = res.x
curves_to_vtk(curves, out_dir / "curves_opt")
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(out_dir / "surf_opt", extra_data=pointData)
