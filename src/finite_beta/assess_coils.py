#!/usr/bin/env python3
import os
import time
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.mhd.vmec import Vmec
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.geo import SurfaceRZFourier
from simsopt.util import proc0_print, comm_world
this_path = os.path.dirname(os.path.abspath(__file__))

QA_or_QH = "QH"
beta = 2.5
filename = 'wout_final.nc'
results_folder = 'results_finally_DMerc'
ncoils = 6
order = 12

nfieldlines = 3 # 10
tmax_fl = 10000 # 20000
degree = 2 # 4

prefix_save = 'optimization'
out_dir_APPENDIX=f"{prefix_save}_{QA_or_QH}_beta{beta:.1f}"
out_dir = os.path.join(this_path,results_folder,QA_or_QH,out_dir_APPENDIX)
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
OUT_DIR = Path("coils")
OUT_DIR.mkdir(parents=True, exist_ok=True)

vmec_file = os.path.join(out_dir,filename)
surf = SurfaceRZFourier.from_wout(filename, nphi=200, ntheta=30, range="full torus")
R_max = np.max(surf.gamma()[0,:,0])
R_axis = np.sum(Vmec(vmec_file).wout.raxis_cc)

proc0_print('Loading coils file')
coils_filename = os.path.join(OUT_DIR,f"biot_savart_nfp{surf.nfp}_{QA_or_QH}_ncoils{ncoils}_order{order}.json")
bs = load(coils_filename)

proc0_print('Computing surface classifier')
surf.to_vtk(OUT_DIR / 'surface_for_Poincare')
sc_fieldline = SurfaceClassifier(surf, h=0.03, p=2)
sc_fieldline.to_vtk(OUT_DIR / 'levelset', h=0.02)

def trace_fieldlines(bfield, label):
    t1 = time.time()
    R0 = np.linspace(R_axis, 0.95*R_max, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/surf.nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-16, comm=comm_world,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        particles_to_vtk(fieldlines_tys, OUT_DIR + f'fieldlines_{label}')
        plot_poincare_data(fieldlines_phi_hits, phis, OUT_DIR + f'poincare_fieldline_{label}.png', dpi=150)


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')

n = 20
rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
zs = surf.gamma()[:, :, 2]
rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/surf.nfp, n*2)
zrange = (0, np.max(zs), n//2)

def skip(rs, phis, zs):
    rphiz = np.asarray([rs, phis, zs]).T.copy()
    dists = sc_fieldline.evaluate_rphiz(rphiz)
    skip = list((dists < -0.05).flatten())
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

