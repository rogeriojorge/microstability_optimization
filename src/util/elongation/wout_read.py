import os

import numpy as np
import scipy.interpolate as spi
from netCDF4 import Dataset

class readWout:
    """
    A class to read wout netCDF files from VMEC

    ...

    Attributes
    ----------
    path : str
        path to the directory in which the wout file is located
    name : str, optional
        name of the wout file being read.  Default is 'wout_HSX_main_opt0.nc'
    space_derivs : bool, optional
        compute fourier amplitudes of flux coordinate differential
        Default is False
    field_derivs : bool, optional
        compute fourier amplitudes of B field vector component differentials
        Default is False

    Methods
    -------
    transForm_3D(s_dom, u_dom, v_dom, ampKeys)
        Performs 3D Fourier transform on specified keys

    transForm_2D_sSec(s_val, u_dom, v_dom, ampKeys)
        Performs 2D Fourier transform along one flux surface on specified keys

    transForm_2D_uSec(s_dom, u_val, v_dom, ampKeys)
        Performs 2D Fourier transform along a poloidal cross section on
        specified keys

    transForm_2D_vSec(s_dom, u_dom, v_val, ampKeys)
        Performs 2D Fourier transform along a toroidal cross section on
        specified keys

    transForm_1D(s_val, u_val, v_val, ampKeys)
        Performs 1D Fourier transform at a particular flux coordinate on
        specified keys

    transForm_listPoints(points, ampKeys):
        Performs 1D Fourier transform at listed flux coordinates on
        specified keys

    calc_metric_tensor(s_dom, u_dom, v_dom):
        Calculate the components of the covariant and contravariant metric tensor.

    Raises
    ------
    IOError
        wout file specified does not exist
    """

    def __init__(self, path, space_derivs=False, field_derivs=False):
        try:
            rootgrp = Dataset(path, 'r')
        except IOError:
            print('File does not exists: '+path)
            raise

        self.space_derivs = space_derivs
        self.field_derivs = field_derivs

        self.nfp = rootgrp['/nfp'][0]
        self.mpol = rootgrp['/mpol'][0]
        self.ntor = rootgrp['/ntor'][0]

        self.ns = rootgrp['/ns'][0]
        self.s_grid = np.linspace(0, 1, self.ns)
        
        self.volume_grid = np.linspace(0, rootgrp['/volume_p'], self.ns)
        self.volume = spi.interp1d(self.s_grid, self.volume_grid)

        self.phi_dom = rootgrp['/phi'][:]
        self.iota = spi.interp1d(self.s_grid, rootgrp['/iotaf'][:])

        self.a_minor = rootgrp['/Aminor_p'][:]
        self.R_major = rootgrp['/Rmajor_p'][:]
        self.r_hat = self.a_minor * np.sqrt(self.s_grid)

        self.xm = rootgrp['/xm'][:]
        self.xn = rootgrp['/xn'][:]
        self.md = len(self.xm)

        self.xm_nyq = rootgrp['/xm_nyq'][:]
        self.xn_nyq = rootgrp['/xn_nyq'][:]
        self.md_nyq = len(self.xm_nyq)

        self.ds = 1. / (self.ns-1)

        self.fourierAmps = {'R': spi.interp1d(self.s_grid, rootgrp['/rmnc'][:, :], axis=0),
                            'Z': spi.interp1d(self.s_grid, rootgrp['/zmns'][:, :], axis=0),
                            'Jacobian': spi.interp1d(self.s_grid, rootgrp['/gmnc'][:, :], axis=0),
                            'Lambda': spi.interp1d(self.s_grid, rootgrp['/lmns'][:, :], axis=0),
                            'Bu_covar': spi.interp1d(self.s_grid, rootgrp['/bsubumnc'][:, :], axis=0),
                            'Bv_covar': spi.interp1d(self.s_grid, rootgrp['/bsubvmnc'][:, :], axis=0),
                            'Bs_covar': spi.interp1d(self.s_grid, rootgrp['/bsubsmns'][:, :], axis=0),
                            'Bu_contra': spi.interp1d(self.s_grid, rootgrp['/bsupumnc'][:, :], axis=0),
                            'Bv_contra': spi.interp1d(self.s_grid, rootgrp['/bsupvmnc'][:, :], axis=0),
                            'Bmod': spi.interp1d(self.s_grid, rootgrp['/bmnc'][:, :], axis=0)
                            }

        if self.md == self.md_nyq:
            self.nyq_limit = False

            self.cosine_keys = ['R', 'Jacobian', 'Bu_covar', 'Bv_covar', 'Bu_contra', 'Bv_contra', 'Bmod']
            self.sine_keys = ['Z', 'Lambda', 'Bs_covar']

        else:
            self.nyq_limit = True

            self.cosine_keys = ['R']
            self.sine_keys = ['Z', 'Lambda']

            self.cosine_nyq_keys = ['Jacobian', 'Bu_covar', 'Bv_covar', 'Bu_contra', 'Bv_contra', 'Bmod']
            self.sine_nyq_keys = ['Bs_covar']

        if space_derivs:
            self.fourierAmps['dR_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/rmnc'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dR_du'] = spi.interp1d(self.s_grid, -rootgrp['/rmnc'][:, :]*self.xm, axis=0)
            self.fourierAmps['dR_dv'] = spi.interp1d(self.s_grid, rootgrp['/rmnc'][:, :]*self.xn, axis=0)

            self.fourierAmps['dZ_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/zmns'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dZ_du'] = spi.interp1d(self.s_grid, rootgrp['/zmns'][:, :]*self.xm, axis=0)
            self.fourierAmps['dZ_dv'] = spi.interp1d(self.s_grid, -rootgrp['/zmns'][:, :]*self.xn, axis=0)

            self.fourierAmps['dL_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/lmns'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dL_du'] = spi.interp1d(self.s_grid, rootgrp['/lmns'][:, :]*self.xm, axis=0)
            self.fourierAmps['dL_dv'] = spi.interp1d(self.s_grid, -rootgrp['/lmns'][:, :]*self.xn, axis=0)

            self.cosine_keys.extend(['dR_ds', 'dZ_du', 'dZ_dv', 'dL_du', 'dL_dv'])
            self.sine_keys.extend(['dR_du', 'dR_dv', 'dZ_ds', 'dL_ds'])

        if field_derivs:
            self.fourierAmps['dBs_du'] = spi.interp1d(self.s_grid, rootgrp['/bsubsmns'][:, :]*self.xm_nyq, axis=0)
            self.fourierAmps['dBs_dv'] = spi.interp1d(self.s_grid, -rootgrp['/bsubsmns'][:, :]*self.xn_nyq, axis=0)

            self.fourierAmps['dBu_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/bsubumnc'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dBu_dv'] = spi.interp1d(self.s_grid, rootgrp['/bsubumnc'][:, :]*self.xn_nyq, axis=0)

            self.fourierAmps['dBv_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/bsubvmnc'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dBv_du'] = spi.interp1d(self.s_grid, -rootgrp['/bsubvmnc'][:, :]*self.xm_nyq, axis=0)

            self.fourierAmps['dBmod_ds'] = spi.interp1d(self.s_grid, np.gradient(rootgrp['/bmnc'][:, :], self.ds, axis=0), axis=0)
            self.fourierAmps['dBmod_du'] = spi.interp1d(self.s_grid, -rootgrp['/bmnc'][:, :]*self.xm_nyq, axis=0)
            self.fourierAmps['dBmod_dv'] = spi.interp1d(self.s_grid, rootgrp['/bmnc'][:, :]*self.xn_nyq, axis=0)

            if self.nyq_limit:
                self.cosine_nyq_keys.extend(['dBs_du', 'dBs_dv', 'dBu_ds', 'dBv_ds', 'dBmod_ds'])
                self.sine_nyq_keys.extend(['dBu_dv', 'dBv_du', 'dBmod_du', 'dBmod_dv'])

            else:
                self.cosine_keys.extend(['dBs_du', 'dBs_dv', 'dBu_ds', 'dBv_ds', 'dBmod_ds'])
                self.sine_keys.extend(['dBu_dv', 'dBv_du', 'dBmod_du', 'dBmod_dv'])

        rootgrp.close()

    def transForm_3D(self, s_dom, u_dom, v_dom, ampKeys):
        """ Performs 3D Fourier transform on specified keys

        Parameters
        ----------
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = int(s_dom.shape[0])
        self.u_num = int(u_dom.shape[0])
        self.v_num = int(v_dom.shape[0])

        self.s_dom = s_dom
        self.u_dom = u_dom
        self.v_dom = v_dom

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(self.xm.reshape(self.md, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_xn = np.dot(self.xn.reshape(self.md, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(self.xm_nyq.reshape(self.md_nyq, 1), pol.reshape(1, self.v_num * self.u_num))
                    tor_nyq_xn = np.dot(self.xn_nyq.reshape(self.md_nyq, 1), tor.reshape(1, self.v_num * self.u_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            fourAmps = self.fourierAmps[key](s_dom)

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_nyq_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_nyq_mu_nv).reshape(self.s_num, self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_sSec(self, s_val, u_dom, v_dom, ampKeys, errKeys=False):
        """ Performs 2D Fourier transform along one flux surface on specified keys

        Parameters
        ----------
        s_val : float
            effective radius of flux surface on which Fourier tronsfrom will
            be performed
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed
        errKeys : bool, optional
            flag to include flux coordinate error propogation. Default is False

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = 1
        self.u_num = int(u_dom.shape[0])
        self.v_num = int(v_dom.shape[0])

        self.s_dom = np.array([s_val])
        self.u_dom = u_dom
        self.v_dom = v_dom

        pol, tor = np.meshgrid(self.u_dom, self.v_dom)

        pol_xm = np.dot(self.xm.reshape(self.md, 1), pol.reshape(1, self.v_num * self.u_num))
        tor_xn = np.dot(self.xn.reshape(self.md, 1), tor.reshape(1, self.v_num * self.u_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(self.xm_nyq.reshape(self.md_nyq, 1), pol.reshape(1, self.v_num * self.u_num))
                    tor_nyq_xn = np.dot(self.xn_nyq.reshape(self.md_nyq, 1), tor.reshape(1, self.v_num * self.u_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            fourAmps = self.fourierAmps[key](s_val)

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_mu_nv).reshape(self.v_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_mu_nv).reshape(self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_nyq_mu_nv).reshape(self.v_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_nyq_mu_nv).reshape(self.v_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_uSec(self, s_dom, u_val, v_dom, ampKeys):
        """ Performs 2D Fourier transform along a poloidal cross section on
        specified keys

        Parameters
        ----------
        s_dom : array
            radial domain on which to perform Fourier transform
        u_val : float
            poloidal coordinate at which to perform Fourier transform
        v_dom : array
            toroidal domain on which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = int(s_dom.shape[0])
        self.u_num = 1
        self.v_num = int(v_dom.shape[0])

        self.s_dom = s_dom
        self.u_dom = np.array([u_val])
        self.v_dom = v_dom

        pol_xm = self.xm.reshape(self.md, 1) * u_val
        tor_xn = np.dot(self.xn.reshape(self.md, 1), self.v_dom.reshape(1, self.v_num))

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = self.xm_nyq.reshape(self.md_nyq, 1) * u_val
                    tor_nyq_xn = np.dot(self.xn_nyq.reshape(self.md_nyq, 1), self.v_dom.reshape(1, self.v_num))

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            fourAmps = self.fourierAmps[key](s_dom)

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_mu_nv).reshape(self.s_num, self.v_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_mu_nv).reshape(self.s_num, self.v_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_nyq_mu_nv).reshape(self.s_num, self.v_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_nyq_mu_nv).reshape(self.s_num, self.v_num)

            else:
                raise NameError('key = {} : is not available'.format(key))

    def transForm_2D_vSec(self, s_dom, u_dom, v_val, ampKeys):
        """ Performs 2D Fourier transform along a toroidal cross section on
        specified keys

        Parameters
        ----------
        s_dom : float
            radial domain on which to perform Fourier transform
        u_dom : array
            poloidal domain on which to perform Fourier transform
        v_val : float
            toroidal coordinate at which to perform Fourier transform
        ampKeys : list
            keys specifying Fourier amplitudes to be transformed

        Raises
        ------
        NameError
            at least on key in ampKeys is not an available Fourier amplitude.
        """
        self.s_num = int(s_dom.shape[0])
        self.u_num = int(u_dom.shape[0])
        self.v_num = 1

        self.s_dom = s_dom
        self.u_dom = u_dom
        self.v_dom = np.array([v_val])

        pol_xm = np.dot(self.xm.reshape(self.md, 1), self.u_dom.reshape(1, self.u_num))
        tor_xn = self.xn.reshape(self.md, 1) * v_val

        cos_mu_nv = np.cos(pol_xm - tor_xn)
        sin_mu_nv = np.sin(pol_xm - tor_xn)

        if self.nyq_limit:
            for key in ampKeys:
                if any(ikey == key for ikey in self.cosine_nyq_keys) or any(ikey == key for ikey in self.sine_nyq_keys):
                    pol_nyq_xm = np.dot(self.xm_nyq.reshape(self.md_nyq, 1), self.u_dom.reshape(1, self.u_num))
                    tor_nyq_xn = self.xn_nyq.reshape(self.md_nyq, 1) * v_val

                    cos_nyq_mu_nv = np.cos(pol_nyq_xm - tor_nyq_xn)
                    sin_nyq_mu_nv = np.sin(pol_nyq_xm - tor_nyq_xn)

        self.invFourAmps = {}
        for key in ampKeys:

            fourAmps = self.fourierAmps[key](s_dom)

            if any(ikey == key for ikey in self.cosine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_mu_nv).reshape(self.s_num, self.u_num)

            elif any(ikey == key for ikey in self.sine_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_mu_nv).reshape(self.s_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.cosine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, cos_nyq_mu_nv).reshape(self.s_num, self.u_num)

            elif self.nyq_limit and any(ikey == key for ikey in self.sine_nyq_keys):
                self.invFourAmps[key] = np.dot(fourAmps, sin_nyq_mu_nv).reshape(self.s_num, self.u_num)

            else:
                raise NameError('key = {} : is not available'.format(key))
