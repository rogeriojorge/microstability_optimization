#!/usr/bin/env python3
import os
import numpy as np
from simsopt import load
from simsopt.mhd import Vmec
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
this_path = os.path.dirname(os.path.abspath(__file__))

ID = '2419171'
vmec_input = 'input.AG'

ntheta = 32
nphi = 32

json_name = f'serial{ID}.json'
surfs, axis, coils = load(os.path.join(this_path, json_name))
boundary = surfs[-1].to_RZFourier()

curves   = [coil._curve for coil in coils]
currents = [coil.current for coil in coils]

base_curves   = [coils[i]._curve for i in range(int(len(curves)/2/boundary.nfp))]
base_currents = [coils[i].current for i in range(int(len(curves)/2/boundary.nfp))]

vmec = Vmec(os.path.join(this_path, vmec_input), ntheta=ntheta, nphi=nphi, range_surface='half period')
vmec.boundary = boundary
vmec.indata.mpol = boundary.mpol+2
vmec.indata.ntor = boundary.ntor+2
vmec.indata.nfp = boundary.nfp
vmec.write_input(f'input.{ID}')

nphi_big   = nphi * 2 * boundary.nfp + 1
ntheta_big = ntheta + 1
quadpoints_theta = np.linspace(0, 1, ntheta_big)
quadpoints_phi   = np.linspace(0, 1, nphi_big)
boundary_big     = SurfaceRZFourier(dofs=boundary.dofs,nfp=boundary.nfp, mpol=boundary.mpol,ntor=boundary.ntor,
                                    quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

curves_to_vtk(curves, 'curves', close=True)
curves_to_vtk(base_curves, 'base_curves', close=True)
boundary.to_vtk('boundary')
boundary_big.to_vtk('boundary_big')