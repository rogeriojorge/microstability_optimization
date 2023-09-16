import netCDF4
import numpy as np

## Calculate growth rate from GS2 output file
def gamma_gs2(gs2File, fractionToConsider = 0.3):
    with netCDF4.Dataset(gs2File,'r') as file2read:
        phi2, t = np.log(file2read['phi2'][()]), file2read['t'][()]
        startIndex = int(len(t)*(1-fractionToConsider))
        data_x, data_y = t[np.isfinite(phi2)], phi2[np.isfinite(phi2)]
        growth_rate = np.polyfit(data_x[startIndex:], data_y[startIndex:], 1)[0] / 2
    return growth_rate

## Calculate quasilinear estimate from GS2 output file
def quasilinear_estimate(gs2File, fractionToConsider=0.3):
    with netCDF4.Dataset(gs2File,'r') as file:
        time, ky, jacob, gds2 = file['t'][()], file['ky'][()], file['jacob'][()], file['gds2'][()]
        phi2_by_ky = np.array(file['phi2_by_ky'][()])
        phi = np.array(file['phi'][()])
        startIndex = int(len(time)*(1-fractionToConsider))

        phi2_by_ky_of_z = phi[:,0,:,0]**2 + phi[:,0,:,1]**2
        growthRates = [np.polyfit(time[np.isfinite(phi2_by_ky[:, i])][startIndex:], 
                                  np.log(phi2_by_ky[np.isfinite(phi2_by_ky[:, i]), i][startIndex:]), 1)[0] / 2 
                       for i in range(len(ky))]

        weighted_kperp2 = np.array([np.sum(ky_each * ky_each * gds2 * phi2_by_ky_of_z[i] * jacob) / 
                                    np.sum(phi2_by_ky_of_z[i] * jacob) for i, ky_each in enumerate(ky)])

    return np.array(growthRates) / weighted_kperp2
