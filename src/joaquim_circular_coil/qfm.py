#!/usr/bin/env python3

import os
import numpy as np
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries, Current, Coil
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, ToroidalFlux, Area, Volume, curves_to_vtk
from simsopt import load

curves=load('/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/circurves_opt.json')
curves_to_vtk(curves, "curves")
currents = [Current(1) * 1e5, Current(-1) * 1e5, Current(1) * 1e5, Current(-1) * 1e5, Current(1) * 1e5, Current(-1) * 1e5]
coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)

mpol = 5
ntor = 5
stellsym = True
nfp = 3
constraint_weight = 1e0
nphi=25
ntheta=25

phis = np.linspace(0, 1, nphi, endpoint=True)
thetas = np.linspace(0, 1, ntheta, endpoint=True)
s = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.set_rc(0, 0, 0.35)
s.set_rc(1, 0, 0.05)
s.set_zs(1, 0, 0.05)
s.set_rc(0, 1, 0.05)
s.set_zs(0, 1, 0.05)

bs.set_points(s.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * s.unitnormal(), axis=2)
pointData = {"B_N": BdotN_surf[:, :, None]}
s.to_vtk("initial_surface", extra_data=pointData)
# exit()

s_volume = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.set_rc(0, 0, 0.35)
s.set_rc(1, 0, 0.1)
s.set_zs(1, 0, 0.1)

qfm = QfmResidual(s, bs)
qfm.J()

vol = Volume(s_volume)
vol_target = vol.J()

qfm_surface = QfmSurface(bs, s, vol, vol_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=1000,
                                                         constraint_weight=constraint_weight)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=1000)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

s.plot()
