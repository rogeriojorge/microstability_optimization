#!/usr/bin/env python3
import os
import sys
import numpy as np
import wout_read as wr
import vecB_tools as vb
from simsopt.mhd import Vmec
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(this_path, '..'))
from qi_functions import MaxElongationPen

def compute_kappa(wout_path):
    wout = wr.readWout(wout_path, space_derivs=True, field_derivs=True)
    
    u_dom = np.linspace(-np.pi, np.pi, 251)
    v_dom = np.linspace(-np.pi, np.pi, 251)
    amp_keys = ['R', 'Z', 'Jacobian', 'Bmod',
                'dR_ds', 'dR_du', 'dR_dv',
                'dZ_ds', 'dZ_du', 'dZ_dv',
                'dBmod_ds', 'dBmod_du', 'dBmod_dv']

    # compute magnetic axis #
    wout.transForm_2D_uSec(np.array([0]), 0, v_dom, ['R', 'Z', 'dR_dv', 'dZ_dv'])
    vmec_ma = wout.invFourAmps
    R_ma = vmec_ma['R'][0]
    Z_ma = vmec_ma['Z'][0]
    g_vv_sqrt = np.sqrt((vmec_ma['dR_dv'][0]**2) + (R_ma**2) + (vmec_ma['dZ_dv'][0]**2))

    # compute flux surface quantities #
    s_val = 0.5
    wout.transForm_2D_sSec(s_val, u_dom, v_dom, amp_keys)
    vmec = wout.invFourAmps
    R_grid = vmec['R']
    Z_grid = vmec['Z']
    B_grid = vmec['Bmod']

    # compute helical basis vectors #
    vecB = vb.BvecTools(wout)
    R_vec, Z_vec = vecB.compute_toroidal_helical_frame()

    # compute poloidally averaged basis vectors #
    X_ma = np.stack((R_ma, Z_ma), axis=1)
    X_surf = np.stack((R_grid, Z_grid), axis=2)
    R_vec_avg, Z_vec_avg = vecB.compute_average_toroidal_helical_frame(R_vec, Z_vec, u_dom)
    
    # compute shaping parameters #
    tor_norm_inv = 1./np.trapz(g_vv_sqrt, v_dom)
    kappa = vecB.compute_shaping_parameters(X_ma, X_surf, R_vec_avg, Z_vec_avg, p_set=[2])
    kappa_avg = np.trapz(kappa[:,0]*g_vv_sqrt, v_dom)*tor_norm_inv

    return kappa_avg

if __name__ == '__main__':
    # wout_path = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'main_coil_0', 'set_1', 'job_0', 'wout_HSX_main_opt0.nc')
    home_directory = os.path.expanduser("~")
    wout_path = f'{home_directory}/local/microstability_optimization/src/finite_beta/results/QH/optimization_QH_beta2.5/wout_final.nc'
    import time
    start=time.time()
    shapes = compute_kappa(wout_path)
    print(f'kappa={shapes} took {time.time()-start}s')
    start=time.time()
    elong = np.array(MaxElongationPen(Vmec(wout_path),t=6.0,ntheta=16,nphi=6,print_all=True))
    print(f'elong={elong} took {time.time()-start}')
