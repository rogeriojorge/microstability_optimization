import os, sys
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator

import wout_read as wr

class BvecTools:
    """
    A class to analyze the magnitude of the magnetic field
    in VMEC flux coordinate geometry

    ...

    Attributes
    ----------
    B_val : array
        3D array of the component of the magnetic field,
        indexed by {s,v,u} coordinates
    s_dom : array
        1D array of the s coordinate domain
    v_dom : array
        1D array of the v coordinate domain
    u_dom : array
        1D array of the u coordinate domain
    s_num : int
        number of points in s domain
    v_num : int
        number of points in v domain
    u_num : int
        number of points in u domain

    Methods
    -------
    interpMagAxis(r, z)
        Interpolates magnetic axis using r, z grid data.

    paraPlot_B(ax, rEff=1)
        Make Parametric plot of the magnetic field along a flux surface.

    plot_B(fig, ax, r, z, v=0, contMan=None)
        Plots poloidal cross sections of the magnetic field.
    """
    def __init__(self, wout):
        # define internal wout object #
        self.wout = wout

        # import number of grid points in each dimension #
        self.s_num = wout.s_num
        self.v_num = wout.v_num
        self.u_num = wout.u_num

        # import dimensional grid #
        self.s_dom = wout.s_dom
        self.v_dom = wout.v_dom
        self.u_dom = wout.u_dom

        # import R and Z grids #
        # self.R_coord = wout.invFourAmps['R']

        # import number of dimensions #
        for key in wout.invFourAmps:
            self.ndim = wout.invFourAmps[key].ndim
            break

    def compute_toroidal_helical_frame(self):
        """ Calculate the helical reference frame where the radial-like
        direction points in the -Grad(B) direction with the toroidal
        component removed, and the vertical-like direction is perpendiucular
        to the radial and toroidal direction.
        """
        try:
            # import radial coordinate and Jacobian #
            R = self.wout.invFourAmps['R']
            jacob = self.wout.invFourAmps['Jacobian']
            a = self.wout.a_minor*np.sqrt(self.s_dom)
            a = np.repeat(a, self.v_num*self.u_num).reshape(R.shape)

            # import radial derivatives #
            dRds = self.wout.invFourAmps['dR_ds']
            dRdu = self.wout.invFourAmps['dR_du']
            dRdv = self.wout.invFourAmps['dR_dv']

            # import vertical derivatives #
            dZds = self.wout.invFourAmps['dZ_ds']
            dZdu = self.wout.invFourAmps['dZ_du']
            dZdv = self.wout.invFourAmps['dZ_dv']

            # import field strength and derivatives #
            Bmod = self.wout.invFourAmps['Bmod']
            dBds = self.wout.invFourAmps['dBmod_ds']
            dBdu = self.wout.invFourAmps['dBmod_du']
            dBdv = self.wout.invFourAmps['dBmod_dv']
        except KeyError:
            raise KeyError('Must include the following list in AmpKeys: '+
                    'R, Jacobian, dR_ds, dR_du, dR_dv, dZ_ds, dZ_du, dZ_dv, '+
                    'Bmod', 'dBmod_ds, dBmod_du, dBmod_dv')

        # compute repeated expresseions #
        R_inv = (1./R)
        B_inv = (1./Bmod)
        jacob_inv = (1./jacob)
        jacob2_inv = jacob_inv**2
        RvZu = dRdv*dZdu-dRdu*dZdv
        RsZv = dRds*dZdv-dRdv*dZds

        # compute contravariant metric components #
        gss = jacob2_inv*(RvZu*RvZu+R*R*(dZdu*dZdu+dRdu*dRdu))
        guu = jacob2_inv*(RsZv*RsZv+R*R*(dZds*dZds+dRds*dRds))
        gvv = R_inv*R_inv
        gsu = jacob2_inv*(RvZu*RsZv-R*R*(dZds*dZdu+dRds*dRdu))
        guv = R_inv*jacob_inv*RsZv
        gvs = R_inv*jacob_inv*RvZu

        # compute covariant metric components #
        g_uv = dRdu*dRdv+dZdu*dZdv
        g_vs = dRdv*dRds+dZdv*dZds

        # compute repeated expression #
        gradBv = dBds*gvs+dBdu*guv+dBdv*gvv

        # compute contravariant components of helical basis vectors #
        Rs = -a*B_inv*(dBds*gss+dBdu*gsu+dBdv*gvs)
        Ru = -a*B_inv*(dBds*gsu+dBdu*guu+dBdv*guv)
        Zs = -(a**2)*B_inv*jacob_inv*(dBdu-gradBv*g_uv)
        Zu = (a**2)*B_inv*jacob_inv*(dBds-gradBv*g_vs)
    
        # compute cylindrical components of helical basis vectors #
        Rr = Rs*dRds+Ru*dRdu
        Rz = Rs*dZds+Ru*dZdu
        Zr = Zs*dRds+Zu*dRdu
        Zz = Zs*dZds+Zu*dZdu

        # normalize helical basis vectors #
        R_norm_inv = 1./np.sqrt(Rr*Rr+Rz*Rz)
        Z_norm_inv = 1./np.sqrt(Zr*Zr+Zz*Zz)
        Rr_norm = Rr*R_norm_inv
        Rz_norm = Rz*R_norm_inv
        Zr_norm = Zr*Z_norm_inv
        Zz_norm = Zz*Z_norm_inv

        # return helical vectors #
        R_vec = np.stack((Rr_norm, Rz_norm), axis=self.ndim)
        Z_vec = np.stack((Zr_norm, Zz_norm), axis=self.ndim)
        return R_vec, Z_vec

    def compute_average_toroidal_helical_frame(self, R_vec, Z_vec, u_dom):
        """ Calculate the poloidally averaged helical basis vectors.

        Parameters
        ----------
        R_vec : 3D arr
            The helical basis vector in the radial-like direction.
        Z_vec : 3D arr
            The helical basis vector in the vertical-like direction.
        u_dom : 1D arr
            Polidal angles over which to perform integration
        """
        # compute average of basis vector components #
        g_uu_sqrt = np.sqrt((self.wout.invFourAmps['dR_du']**2) + (self.wout.invFourAmps['dZ_du']**2))
        theta_norm_inv = 1./np.trapz(g_uu_sqrt, u_dom, axis=1)
        R1 = np.trapz(R_vec[:,:,0]*g_uu_sqrt, u_dom, axis=1)*theta_norm_inv
        R2 = np.trapz(R_vec[:,:,1]*g_uu_sqrt, u_dom, axis=1)*theta_norm_inv
        Z1 = np.trapz(Z_vec[:,:,0]*g_uu_sqrt, u_dom, axis=1)*theta_norm_inv
        Z2 = np.trapz(Z_vec[:,:,1]*g_uu_sqrt, u_dom, axis=1)*theta_norm_inv

        # normalize basis vector components #
        R_norm_inv = 1./np.sqrt(R1*R1+R2*R2)
        Z_norm_inv = 1./np.sqrt(Z1*Z1+Z2*Z2)
        R1_norm = R1*R_norm_inv
        R2_norm = R2*R_norm_inv
        Z1_norm = Z1*Z_norm_inv
        Z2_norm = Z2*Z_norm_inv

        # return averaged basis vectors #
        R_avg_vec = np.stack((R1_norm, R2_norm), axis=1)
        Z_avg_vec = np.stack((Z1_norm, Z2_norm), axis=1)
        return R_avg_vec, Z_avg_vec

    def compute_shaping_parameters(self, X_ma, X_surf, R_vec, Z_vec, p_set=[2,3,4]):
        """ Compute the shaping parameters in the provided cross sections. Shaping parameters
        are returned across all provided cross sections.

        Parameters
        ----------
        X_ma : 2D arr
            Magnetic axis is cylindrical coordinates. The first axis is over toroidal angle.
            The second axis provide R, Z points.
        X_surf : 3D arr
            Flux surface in each toroidal cross section. The first axis is over toroidal angle.
            The second axis if over the poloidal angle. The third axis provides R, Z points.
        R_vec : 2D arr
            Helical radial-like basis vector. The first axis is over toroidal angle. The second
            axis provides R, Z points.
        Z_vec : 2D arr
            Helical vertical-like basis vector. The first axis is over toroidal angle. The second
            axis provides R, Z points.
        p_set : list, optional
            A list of shaping parameters to evaluate. Default is [2, 3, 4], which corresponds to
            elongation, triangularity, and squareness, respectively.
        """
        # define minor radius of flux surfaces #
        X_ma = np.repeat(X_ma, X_surf.shape[1], axis=0).reshape(X_ma.shape[0], X_surf.shape[1], 2)
        X = X_surf-X_ma
        X_norm = np.linalg.norm(X, axis=2)

        # compute theta mapping #
        R_vec = np.repeat(R_vec, X_surf.shape[1], axis=0).reshape(R_vec.shape[0], X_surf.shape[1], 2)
        Z_vec = np.repeat(Z_vec, X_surf.shape[1], axis=0).reshape(Z_vec.shape[0], X_surf.shape[1], 2)
        X_dot_Z = np.sum(X*Z_vec, axis=2)
        X_dot_R = np.sum(X*R_vec, axis=2)
        theta = np.arctan2(X_dot_Z, X_dot_R)
        shaping_parameters = np.empty((theta.shape[0], len(p_set)))
        for i, the in enumerate(theta):
            the_map = np.stack((the, X_norm[i]), axis=1)
            the_sort = np.array(sorted(the_map, key=lambda x: x[0]))
            the = the_sort[:,0]
            X2_the = the_sort[:,1]**2
            X2_norm_inv = 1./np.trapz(X2_the, the)
            for j, p in enumerate(p_set):
                shape = ((-1)**(p-1))*np.trapz(X2_the*np.cos(p*the), the)*X2_norm_inv
                shaping_parameters[i,j] = shape

        return shaping_parameters
