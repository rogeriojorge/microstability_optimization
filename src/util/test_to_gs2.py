#!/usr/bin/env python3
import os
import time
import netCDF4
import subprocess
import numpy as np
from pathlib import Path
from simsopt.mhd import Vmec
import matplotlib.pyplot as plt
from simsopt._core.util import Struct
# from simsopt.mhd.vmec_diagnostics import to_gs2, vmec_fieldlines
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
from to_gs2_function import to_gs2
this_path = Path(__file__).parent.resolve()

gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2'
gs2_input_name = 'gs2Input.in'
gridout_file = os.path.join(this_path,"grid.out")
gridout_test_file = os.path.join(this_path,"test_gridout.dat")
vmec_filename = os.path.join(this_path,'wout_test.nc')#'wout_nfp4_QH_000_000000.nc'#'input.nfp4_QH'
s_radius = 0.6354167
alpha_fieldline = 0
phi = np.linspace(-5*np.pi/3, 5*np.pi/3, 201)
nlambda = 20

vmec = Vmec(vmec_filename)

grid_struct = vmec_fieldlines(vmec, s_radius, alpha_fieldline, phi1d=phi)
test_grid_struct = Struct()
# test_grid_dict = {}
with open(gridout_test_file) as fp:
    for line_number, line in enumerate(fp):
        if line_number<6: continue
        if np.mod(line_number,2)==0:
            this_line_array = line.strip()
        else:
            numbers = []
            for item in line.split():
                numbers.append(float(item))
            # test_grid_dict[this_line_array] = np.array(numbers)
            test_grid_struct.__setattr__(this_line_array, np.array(numbers))

variables = ['zeta', 'bmag', 'gradpar', 'gds2', 'gds21', 'gds22', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0']
for variable1 in variables:
    if variable1=='zeta': variable2='phi'
    elif variable1=='gradpar': variable2='gradpar_phi'
    elif variable1=='gradpar': variable2='gradpar_phi'
    else: variable2 = variable1

    if variable1=='gbdrift' or variable1=='cvdrift':
        try: np.testing.assert_allclose(getattr(test_grid_struct, variable1),-np.ravel(getattr(grid_struct, variable2)), atol=5e-4, rtol=4e-3)
        except Exception as e:
            print(variable1)
            print(e)
            exit()
    else:
        try: np.testing.assert_allclose(getattr(test_grid_struct, variable1), np.ravel(getattr(grid_struct, variable2)), atol=5e-4, rtol=4e-3)
        except Exception as e:
            print(variable1)
            print(e)
            exit()
    # print(f'{variable1} is ok')

# Run GS2 and analyze results
phi = np.linspace(-2*np.pi, 2*np.pi, 51)

start_time = time.time()
to_gs2(gridout_file, vmec, s_radius, alpha_fieldline, phi1d=phi, nlambda=nlambda)
print('Time to run to_gs2 =',time.time()-start_time,'s')

print('Running GS2 VMEC...')
start_time = time.time()
f_log = os.path.join(this_path,"gs2.log")
bashCommand = f"{gs2_executable} {os.path.join(this_path,gs2_input_name)}"
with open(f_log, 'w') as fp:
    p = subprocess.Popen(bashCommand.split(),stdout=fp)
p.wait()
print('Time to run gs2 =',time.time()-start_time,'s')

fractionToConsider = 0.3 # fraction of time from the simulation period to consider

file2read = netCDF4.Dataset(os.path.join(this_path,gs2_input_name[:-3]+".out.nc"),'r')
tX = file2read.variables['t'][()]
kyX  = file2read.variables['ky'][()]
phi2_by_kyX  = file2read.variables['phi2_by_ky'][()]
omegaX  = file2read.variables['omega'][()]
qparflux2_by_ky = file2read.variables['qparflux2_by_ky'][()]

startIndexX  = int(len(tX)*(1-fractionToConsider))

growthRateX  = []
for i in range(len(kyX)):
    maskX  = np.isfinite(phi2_by_kyX[:,i])
    data_xX = tX[maskX]
    data_yX = phi2_by_kyX[maskX,i]
    fitX  = np.polyfit(data_xX[startIndexX:], np.log(data_yX[startIndexX:]), 1)
    thisGrowthRateX  = fitX[0]/2
    growthRateX.append(thisGrowthRateX)

realFreqVsTimeX  = []
realFrequencyX   = []
qparVsTimeX = []
qparX = []
for i in range(len(kyX)):
    realFreqVsTimeX.append(omegaX[:,i,0,0])
    realFrequencyX.append(np.mean(realFreqVsTimeX[i][startIndexX:]))
    qparVsTimeX.append(qparflux2_by_ky[:,0,i])
    qparX.append(np.mean(qparVsTimeX[i][startIndexX:]))

numRows = 1
numCols = 3

plt.subplot(numRows, numCols, 1)
plt.plot(kyX,growthRateX,'.-',label=r'VMEC')
plt.xlabel(r'$k_y$')
plt.ylabel(r'$\gamma$')
plt.xscale('log')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
# plt.legend(frameon=False,prop=dict(size='xx-small'),loc=0)

plt.subplot(numRows, numCols, 2)
plt.plot(kyX,realFrequencyX,'.-',label=r'VMEC')
plt.xlabel(r'$k_y$')
plt.ylabel(r'$\omega$')
plt.xscale('log')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
# plt.legend(frameon=False,prop=dict(size=12),loc=0)

plt.subplot(numRows, numCols, 3)
plt.plot(kyX,qparX,'.-',label=r'VMEC')
plt.xlabel(r'$k_y$')
plt.ylabel(r'$q_{par}$')
plt.xscale('log')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
# plt.legend(frameon=False,prop=dict(size=12),loc=0)

plt.tight_layout()
#plt.subplots_adjust(left=0.14, bottom=0.15, right=0.98, top=0.96)
# plt.savefig(runName[:-3]+"_GammaOmegaKy.pdf", format='pdf')
plt.show()

#### REMOVE SPURIOUS FILES
appendices_to_remove = ['amoments','eigenfunc','error','exit_reason','fields','g','lpc','mom2','moments','out','used_inputs.in','vres','vres2','vspace_integration_error']
for appendix in appendices_to_remove:
    os.remove(f'{gs2_input_name[:-3]}.{appendix}')