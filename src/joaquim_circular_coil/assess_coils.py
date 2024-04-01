#!/usr/bin/env python3
import os
import re
import time
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd.vmec import Vmec
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk, 
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data,
                           coils_to_focus, coils_to_makegrid, Current, Coil, BiotSavart)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import proc0_print, comm_world
this_path = os.path.dirname(os.path.abspath(__file__))

curves=load('/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/circurves_opt.json')
currents = [Current(-1) * 1e5 for c in curves]
coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
bs = BiotSavart(coils)

nfieldlines = 24
tmax_fl = 14000 # 20000
degree = 4
extend_distance = 0.1 # 0.2
nfieldlines_to_plot = 10

interpolate_field = True

proc0_print('Computing surface classifier')
# surf.to_vtk(os.path.join(OUT_DIR,'surface_for_Poincare'))
sc_fieldline = SurfaceClassifier(surf, h=0.03*R_axis, p=2)
# sc_fieldline.to_vtk(os.path.join(OUT_DIR,'levelset'), h=0.04*R_axis)

def trace_fieldlines(bfield, label):
    t1 = time.time()
    R0 = np.linspace(0.999*R_axis, 1.00*R_max, nfieldlines)
    proc0_print(f"R0={R0}", flush=True)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/surf.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        for i, fieldline_tys in enumerate(fieldlines_tys[-nfieldlines_to_plot:]):
            particles_to_vtk([fieldline_tys], os.path.join(OUT_DIR,f'fieldlines_{label}_{i}'))
        # particles_to_vtk(fieldlines_tys[-6:], os.path.join(OUT_DIR,f'fieldlines_{label}'))
        plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR,f'poincare_fieldline_{label}.png'), dpi=300, s=1.5, surf=surf_vmec)

if interpolate_field:
    n = 30
    rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
    zs = surf.gamma()[:, :, 2]
    rrange = (0.8*np.min(rs), 1.2*np.max(rs), n)
    phirange = (0, 2*np.pi/surf.nfp, n*2)
    zrange = (0, 1.2*np.max(np.abs(zs)), n//2)

    def skip(rs, phis, zs):
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05*R_axis*2.0).flatten())
        # skip = [False]*len(skip)
        proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip

    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=surf.nfp, stellsym=True, skip=skip)
    proc0_print('Done initializing InterpolatedField.')

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))

    proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))

    proc0_print('Beginning field line tracing')
    trace_fieldlines(bsh, 'bsh')
else:
    trace_fieldlines(bs, 'bs')
