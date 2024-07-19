#!/usr/bin/env python3
import os
import numpy as np
from simsopt import load
from simsopt.mhd import Vmec, Boozer
import matplotlib.pyplot as plt
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, curves_to_vtk
from simsopt.field import BiotSavart
import sys
sys.path.append("..")
from util import vmecPlot2, booz_plot
this_path = os.path.dirname(os.path.abspath(__file__))

ids = ['2419171','1555109','2091653','2368127','2368027']
vmec_input = 'input.AG'
ntheta = 32
nphi = 32

for ID in ids:
    print('Running ID:', ID)
    results_path = os.path.join(this_path, ID)
    os.makedirs(ID, exist_ok=True)
    os.chdir(results_path)

    json_name = f'serial{ID}.json'
    surfs, axis, coils = load(os.path.join(this_path, json_name))
    if ID == '2419171': surfs = load(os.path.join(this_path, 'serial2419171_higherres.json'))
    
    surf = surfs[-1]

    # new_surf = SurfaceXYZTensorFourier(nfp=surf.nfp, mpol=surf.mpol, ntor=surf.ntor, dofs=surf.dofs,
    #                                    quadpoints_phi=np.linspace(0,1/2/surf.nfp, nphi, endpoint=False),
    #                                    quadpoints_theta=np.linspace(0,1,ntheta, endpoint=False))
    # boundary = new_surf.to_RZFourier()
    boundary = surf.to_RZFourier()

    curves   = [coil._curve for coil in coils]
    currents = [coil.current for coil in coils]

    base_curves   = [coils[i]._curve for i in range(int(len(curves)/2/boundary.nfp))]
    base_currents = [coils[i].current for i in range(int(len(curves)/2/boundary.nfp))]

    bs = BiotSavart(coils)

    nphi_big   = nphi * 2 * boundary.nfp + 1
    ntheta_big = ntheta + 1
    quadpoints_theta = np.linspace(0, 1, ntheta_big)
    quadpoints_phi   = np.linspace(0, 1, nphi_big)
    boundary_big     = SurfaceRZFourier(dofs=boundary.dofs,nfp=boundary.nfp, mpol=boundary.mpol,ntor=boundary.ntor,
                                        quadpoints_phi=quadpoints_phi,quadpoints_theta=quadpoints_theta)

    # cross_section = boundary_big.cross_section(phi=0.5)
    # r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    # z_interp = cross_section[:, 2]
    # plt.plot(r_interp, z_interp, linewidth=1, c='k')
    # plt.show()
    # exit()

    curves_to_vtk(curves, 'curves', close=True)
    curves_to_vtk(base_curves, 'base_curves', close=True)
    
    nphi = len(boundary.quadpoints_phi)
    ntheta = len(boundary.quadpoints_theta)
    bs.set_points(boundary.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = (np.sum(Bbs * boundary.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi,ntheta,1))
    boundary.to_vtk('boundary', extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    
    # bs.set_points(new_surf.gamma().reshape((-1, 3)))
    # Bbs = bs.B().reshape((nphi, ntheta, 3))
    # BdotN = (np.sum(Bbs * new_surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    # Bmod = bs.AbsB().reshape((nphi,ntheta,1))
    # new_surf.to_vtk('boundary_original_more_points')
    
    ntheta_original = len(surf.quadpoints_theta)
    nphi_original = len(surf.quadpoints_theta)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_original, ntheta_original, 3))
    BdotN = (np.sum(Bbs * surf.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi_original,ntheta_original,1))
    surf.to_vtk('boundary_original', extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})
    
    bs.set_points(boundary_big.gamma().reshape((-1, 3)))
    Bbs = bs.B().reshape((nphi_big, ntheta_big, 3))
    BdotN = (np.sum(Bbs * boundary_big.unitnormal(), axis=2)) / np.linalg.norm(Bbs, axis=2)
    Bmod = bs.AbsB().reshape((nphi_big,ntheta_big,1))
    boundary_big.to_vtk('boundary_big', extra_data= {"B.n/B": BdotN[:, :, None], "B": Bmod})

    try:
        vmec = Vmec(os.path.join(this_path, vmec_input), ntheta=ntheta, nphi=nphi, range_surface='field period')
        vmec.boundary = boundary
        vmec.indata.mpol = boundary.mpol
        vmec.indata.ntor = boundary.ntor
        vmec.indata.nfp = boundary.nfp
        vmec.write_input(f'input.{ID}')
        vmec.run()
    except Exception as e:
        print(e)
        try:
            boundary.change_resolution(8,8)
            vmec.boundary = boundary
            vmec.indata.mpol = boundary.mpol
            vmec.indata.ntor = boundary.ntor
            vmec.indata.nfp = boundary.nfp
            vmec.write_input(f'input.{ID}')
            vmec.run()
        except Exception as e:
            try:
                boundary.change_resolution(5,5)
                vmec.boundary = boundary
                vmec.indata.mpol = boundary.mpol
                vmec.indata.ntor = boundary.ntor
                vmec.indata.nfp = boundary.nfp
                vmec.write_input(f'input.{ID}')
                vmec.run()
            except:
                boundary.change_resolution(2,2)
                vmec.boundary = boundary
                vmec.indata.mpol = boundary.mpol
                vmec.indata.ntor = boundary.ntor
                vmec.indata.nfp = boundary.nfp
                vmec.write_input(f'input.{ID}')
                vmec.run()

    try: vmecPlot2.main(file=vmec.output_file)
    except Exception as e: print(e)
    try: booz_plot.main(file=vmec.output_file)
    except Exception as e: print(e)
    
    os.chdir(this_path)