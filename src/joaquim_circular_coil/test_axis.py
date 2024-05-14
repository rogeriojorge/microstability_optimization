import numpy as np
import matplotlib.pyplot as plt

# Fourier coefficients
rc_n = [0.389270263078912, -0.00433461050975322, -0.00538230780673345, 
        0.0482252059477651, -0.00212640878699802, -0.00122539478265202, 
        -0.0137482105891103, -4.30695420898079e-05, 0.000260414753336204, 
        -0.00130868123191214]

zs_n = [-0, 0.000656234050922838, -0.00061762403519379, 
        0.0185307042696452, 5.9373111195823e-05, -0.000318073160137614, 
        0.000114805449047029, -6.51042714938295e-05, -4.70008097035017e-06, 
        -0.000699608032613906]

# Parameters
nphi = 1000  # Number of points in phi
phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
d_phi = phi[1] - phi[0]

# Initialize arrays
R0 = np.zeros(nphi)
Z0 = np.zeros(nphi)
R0p = np.zeros(nphi)
Z0p = np.zeros(nphi)
R0pp = np.zeros(nphi)
Z0pp = np.zeros(nphi)
R0ppp = np.zeros(nphi)
Z0ppp = np.zeros(nphi)

# Compute R, Z, and their derivatives
for jn in range(len(rc_n)):
    n = jn
    sinangle = np.sin(n * phi)
    cosangle = np.cos(n * phi)
    R0 += rc_n[jn] * cosangle
    Z0 += zs_n[jn] * sinangle
    R0p += rc_n[jn] * (-n * sinangle)
    Z0p += zs_n[jn] * (n * cosangle)
    R0pp += rc_n[jn] * (-n * n * cosangle)
    Z0pp += zs_n[jn] * (-n * n * sinangle)
    R0ppp += rc_n[jn] * (n * n * n * sinangle)
    Z0ppp += zs_n[jn] * (-n * n * n * cosangle)

# Compute differential arc length
d_l_d_phi = np.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi

# Compute tangent and normal vectors
d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()

tangent_cylindrical = d_r_d_phi_cylindrical / d_l_d_phi[:, None]
d_tangent_d_l_cylindrical = (-d_r_d_phi_cylindrical * d2_l_d_phi2[:, None] / d_l_d_phi[:, None] \
                            + d2_r_d_phi2_cylindrical) / (d_l_d_phi[:, None] * d_l_d_phi[:, None])

# Compute curvature
curvature = np.sqrt(np.sum(d_tangent_d_l_cylindrical**2, axis=1))

# Compute normal vector
normal_cylindrical = d_tangent_d_l_cylindrical / curvature[:, None]

# Compute binormal vector
binormal_cylindrical = np.cross(tangent_cylindrical, normal_cylindrical)

# Compute torsion
d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

cross_product = np.cross(d2_r_d_phi2_cylindrical, d3_r_d_phi3_cylindrical)
torsion_numerator = np.einsum('ij,ij->i', d_r_d_phi_cylindrical, cross_product)
torsion_denominator = np.einsum('ij,ij->i', np.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical), 
                                np.cross(d_r_d_phi_cylindrical, d2_r_d_phi2_cylindrical))
torsion = torsion_numerator / torsion_denominator

# Plot curvature and torsion
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(phi, curvature, label='Curvature')
plt.xlabel('phi')
plt.ylabel('Curvature')
plt.title('Curvature vs Phi')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(phi, torsion, label='Torsion', color='orange')
plt.xlabel('phi')
plt.ylabel('Torsion')
plt.title('Torsion vs Phi')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
