#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec, vmec_fieldlines
this_path = os.path.dirname(os.path.abspath(__file__))

min_theta = -2*np.pi # minimum field line angle
max_theta = 2*np.pi  # maximum field line angle
ntheta = 250        # number of points along the field line
surface = 0.5      # radial position of the surface between 0 and 1
alpha = 0          # field line label such that theta_pest=alpha+iota*phi
# vmec_input_file = 'input.nfp4_QH'
# vmec_output_file = 'wout_nfp4_QH.nc'
vmec_output_file = 'wout_nfp2_QA.nc'

v = Vmec(os.path.join(this_path,vmec_output_file))
theta = np.linspace(min_theta, max_theta, ntheta)
fl = vmec_fieldlines(v, surface, alpha, theta1d=theta, plot=False)

plt.figure(figsize=(12, 9))
nrows = 5
ncols = 3
variables = ['modB', 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_psi',
             'B_cross_kappa_dot_grad_alpha', 'B_cross_kappa_dot_grad_psi',
             'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
             'gradpar_phi', 'gbdrift', 'gbdrift0', 'cvdrift', 'gds2', 'gds21', 'gds22']

# L_{ref} = Aminor_p
# B_{ref} = 2 * abs(edge_toroidal_flux_over_2pi) / (Lref * Lref)
# \hat{s} = (-2 * s / iota) * d_iota_d_s
# sign_{psi} = np.sign(edge_toroidal_flux_over_2pi)
titles =    [r'$|\vec{B}|$',r'$\vec{B}\times\nabla|\vec{B}|\cdot\nabla\alpha$',r'$\vec{B}\times\nabla|\vec{B}|\cdot\nabla\psi$',
             r'$\vec{B}\times\vec{\kappa}\cdot\nabla\alpha$',r'$\vec{B}\times\vec{\kappa}\cdot\nabla\psi$',
             r'$|\nabla\alpha|^2$',r'$\nabla\alpha\cdot\nabla\psi$',r'$|\nabla\psi|^2$',
             r'$\vec{b}\cdot\nabla\phi$',
             r'gbdrift = $-2 B_{ref} L_{ref}^2 \frac{sign_{\psi}\sqrt{s}}{B^3} (\vec{B}\times\nabla|\vec{B}|\cdot\nabla\alpha) $',
             r'gbdrift0 = $(\vec{B}\times\nabla|\vec{B}|\cdot\nabla\psi) \frac{2 sign_{\psi} \hat{s}}{B^3 \sqrt{s}} $',
             r"cvdrift = gbdrift - $2 \sqrt{s} B_{ref} L_{ref}^2 \frac{\mu_0 p'(s) sign_{\psi}}{\frac{-phiedge}{2 \pi} |\vec{B}|^2}$",
             r'gds2 = $L_{ref}^2 s |\nabla\alpha|^2$',r'gds21 = $\frac{\hat{s}}{B_{ref}}(\nabla\alpha\cdot\nabla\psi)$',
             r'gds22 = $ \frac{\hat{s}^2}{L_{ref}^2 B_{ref}^2 s} |\nabla\psi|^2$']

for j, (variable, title) in enumerate(zip(variables, titles)):
    plt.subplot(nrows, ncols, j + 1)
    plt.plot(fl.phi[0, 0, :], eval("fl." + variable + '[0, 0, :]'))
    plt.xlabel('Standard toroidal angle $\phi$')
    plt.title(title)

plt.figtext(0.5, 0.995, f'surface s={surface}, field line alpha={alpha}', ha='center', va='top')
plt.tight_layout()
plt.show()