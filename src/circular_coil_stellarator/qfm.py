#!/usr/bin/env python3

import os
import re
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd.vmec import Vmec
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, ToroidalFlux, Area, Volume
this_path = os.path.dirname(os.path.abspath(__file__))

filename_wout = f'wout_final.nc'
filename_input = f'input.final'
results_folder = f'optimization_QA_ncoils4_nonplanar_symcoils'
coils_file = f'biot_savart_maxmode3.json'
ncoils = int(re.search(r'ncoils(\d+)', results_folder).group(1))

mpol = 8
ntor = 8
constraint_weight = 1e0
extend_distance = 0.01
tol = 1e-15
maxiter = 1000

out_dir = os.path.join(this_path,results_folder)
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
OUT_DIR = Path("coils")
OUT_DIR.mkdir(parents=True, exist_ok=True)
vmec_file_input = os.path.join(out_dir,filename_input)
surf = SurfaceRZFourier.from_vmec_input(vmec_file_input, nphi=200, ntheta=30, range="full torus")

print('Loading coils file')
coils_filename = os.path.join(OUT_DIR,coils_file)
bs = load(coils_filename)

coils = bs.coils
base_curves = [coils[i]._curve for i in range(ncoils)]
base_currents = [coils[i]._current for i in range(ncoils)]

stellsym = True
nfp = surf.nfp

print('Creating surface object')
phis = np.linspace(0, 1/nfp, 25, endpoint=False)
thetas = np.linspace(0, 1, 25, endpoint=False)
s = SurfaceRZFourier(dofs = surf.dofs, mpol=surf.mpol, ntor=surf.ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.change_resolution(mpol, ntor)
s.extend_via_normal(extend_distance)
# First optimize at fixed volume

qfm = QfmResidual(s, bs)
qfm.J()

vol = Volume(s)
vol_target = vol.J()

qfm_surface = QfmSurface(bs, s, vol, vol_target)
print(f"Initial ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=maxiter, constraint_weight=constraint_weight)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=maxiter)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
# Check that volume is not changed
print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

vmec_QFM = Vmec(vmec_file_input, verbose=True)
vmec_QFM.indata.mpol = mpol
vmec_QFM.indata.ntor = ntor
vmec_QFM.boundary = s
vmec_QFM.indata.ns_array[:3]    = [  16]#,    51,    101]
vmec_QFM.indata.niter_array[:3] = [ 2000]#,  3000, 20000]
vmec_QFM.indata.ftol_array[:3]  = [1e-12]#, 1e-14, 1e-14]
vmec_QFM.indata.am[0:10] = [0]*10
vmec_QFM.write_input(os.path.join(OUT_DIR,f'input.qfm'))
vmec_QFM.run()
print(f'iota from vmec = {vmec_QFM.wout.iotaf}')