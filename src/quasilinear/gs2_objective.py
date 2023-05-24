## calculate_gamma_gs2.py
# Description: Calculate the growth rate using GS2
import os
import glob
import shutil
import netCDF4
import subprocess
import numpy as np
from simsopt.mhd import Vmec
from ..util.replace import replace
from ..util.to_gs2  import to_gs2
from ...inputs import (gs2_executable, s_radius, alpha_fieldline,
                       LN, LT, nlambda, nstep, dt, aky_min, aky_max,
                       naky, ngauss, negrid, phi_GS2, GROWTHRATE_THRESHOLD)

replace_patterns = [
    (' nstep = 150', f' nstep = {nstep}'),
    (' delt = 0.4 ! Time step', f' delt = {dt} ! Time step'),
    (' fprim = 1.0 ! -1/n (dn/drho)', f' fprim = {LN} ! -1/n (dn/drho)'),
    (' tprim = 3.0 ! -1/T (dT/drho)', f' tprim = {LT} ! -1/T (dT/drho)'),
    (' aky_min = 0.4', f' aky_min = {aky_min}'),
    (' aky_max = 5.0', f' aky_max = {aky_max}'),
    (' naky = 4', f' naky = {naky}'),
    (' ngauss = 3 ! Number of untrapped pitch-angles moving in one direction along field line.',
        f' ngauss = {ngauss} ! Number of untrapped pitch-angles moving in one direction along field line.'),
    (' negrid = 10 ! Total number of energy grid points',
        f' negrid = {negrid} ! Total number of energy grid points')
]

## Calculate growth rate from GS2 output file
def gamma_gs2(gs2File, fractionToConsider = 0.3):
    with netCDF4.Dataset(gs2File,'r') as file:
        phi2, t = np.log(file['phi2'][()]), file['t'][()]
        startIndex = int(len(t)*(1-fractionToConsider))
        data_x, data_y = t[np.isfinite(phi2)], phi2[np.isfinite(phi2)]
        return np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)[0] / 2

## Calculate quasilinear estimate from GS2 output file
def quasilinear_estimate(gs2File, fractionToConsider=0.3):
    with netCDF4.Dataset(gs2File,'r') as file:
        time, ky, jacob, gds2 = file['t'][()], file['ky'][()], file['jacob'][()], file['gds2'][()]
        phi2_by_ky, phi = np.array(file['phi2_by_ky'][()]), np.array(file['phi'][()])
        startIndex, phi2_by_ky_of_z = int(len(time)*(1-fractionToConsider)), phi[:,0,:,0]**2 + phi[:,0,:,1]**2
        growthRates = [np.polyfit(time[np.isfinite(phi2_by_ky[:, i])][startIndex:], np.log(phi2_by_ky[np.isfinite(phi2_by_ky[:, i]), i][startIndex:]), 1)[0] / 2 for i in range(len(ky))]
        weighted_kperp2 = np.array([np.sum(ky_each * ky_each * gds2 * phi2_by_ky_of_z[i] * jacob) / np.sum(phi2_by_ky_of_z[i] * jacob) for i, ky_each in enumerate(ky)])
        return np.array(growthRates) / weighted_kperp2 / naky

## Get objetctive function for GS2
def get_gs2_objective(v: Vmec, OUT_DIR, quasilinear=True):
    gs2_input_name = "gs2-{}".format(v.output_file.split('/')[-1][5:-3])
    gs2_input_file = os.path.join(OUT_DIR, f'{gs2_input_name}.in')
    replace_patterns.append((' gridout_file = "grid.out"', f' gridout_file = "grid_{gs2_input_name}.out"'))
    try:
        v.run()
        shutil.copy(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'gs2Input.in'), gs2_input_file)
        for pattern, subst in replace_patterns:
            replace(gs2_input_file, pattern, subst)
        to_gs2(os.path.join(OUT_DIR, f'grid_{gs2_input_name}.out'), v, s_radius, alpha_fieldline, phi1d=phi_GS2, nlambda=nlambda)
        bashCommand = f"{gs2_executable} {gs2_input_file}"
        p = subprocess.Popen(bashCommand.split(), stderr=subprocess.STDOUT, stdout=subprocess.DEVNULL)
        p.wait()
        gs2_output = os.path.join(OUT_DIR, f"{gs2_input_name}.out")
        objective = quasilinear_estimate(gs2_output) if quasilinear else gamma_gs2(gs2_output)
        if not np.isfinite(objective): 
            objective = GROWTHRATE_THRESHOLD
    except: objective = GROWTHRATE_THRESHOLD
    for file in glob.glob(os.path.join(OUT_DIR, f"*{gs2_input_name}*")) + glob.glob(os.path.join(OUT_DIR, f".{gs2_input_name}*")):
        try: os.remove(file)
        except: pass
    return objective
