import numpy as np
from simsopt.mhd.vmec_diagnostics import vmec_fieldlines
def to_gs2(filename, vs, s=0.5, alpha=0, nlambda=30, theta1d=None, phi1d=None, phi_center=0):
    r"""
    Compute field lines and geometric quantities along the
    field lines in a vmec configuration needed to run the
    gyrokinetic GS2 code.

    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        alpha: Values of the field line label :math:`\alpha` on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta1d: 1D array of :math:`\theta_{pest}` values, setting the grid points
          along the field line and the parallel extent of the field line.
        phi1d: 1D array of :math:`\phi` values, setting the grid points along the
          field line and the parallel extent of the field line.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
    """
    try: assert not isinstance(alpha, (list, tuple, np.ndarray))
    except Exception: raise ValueError("Only working for a single field line, alpha should be a scalar quantity")
    arrays = vmec_fieldlines(vs, s, alpha, theta1d=theta1d, phi1d=phi1d, phi_center=phi_center, plot=False, show=True)
    nperiod = 1
    drhodpsi = 1.0
    rmaj     = 1.0
    kxfac    = 1.0
    shat = arrays.shat[0]
    q = 1/arrays.iota[0]
    phi = arrays.phi[0,0]
    ntheta=len(phi)-1
    ntgrid=int(np.floor(len(phi)/2))
    bMax=np.max(arrays.bmag[0])
    bMin=np.min(arrays.bmag[0])
    with open(filename,'w') as f:
        f.write(f"nlambda\n{nlambda}\nlambda")
        for i in range(nlambda):
            f.write(f"\n{(bMax - bMax*i + bMin*i - bMin*nlambda)/(bMax*bMin - bMax*bMin*nlambda)}")
        f.write("\nntgrid nperiod ntheta drhodpsi rmaj shat kxfac q")
        f.write(f"\n{ntgrid} {nperiod} {ntheta} {drhodpsi} {rmaj} {shat} {kxfac} {q}")
        f.write("\ngbdrift gradpar grho tgrid")
        for gbdrift, gradpar, tgrid in zip(arrays.gbdrift[0,0], arrays.gradpar_phi[0,0], phi):
            f.write(f"\n{gbdrift} {gradpar} 1.0 {tgrid}")
        f.write("\ncvdrift gds2 bmag tgrid")
        for cvdrift, gds2, bmag, tgrid in zip(arrays.cvdrift[0,0], arrays.gds2[0,0], arrays.bmag[0,0], phi):
            f.write(f"\n{cvdrift} {gds2} {bmag} {tgrid}")
        f.write("\ngds21 gds22 tgrid")
        for gds21, gds22, tgrid in zip(arrays.gds21[0,0], arrays.gds22[0,0], phi):
            f.write(f"\n{gds21} {gds22} {tgrid}")
        f.write("\ncvdrift0 gbdrift0 tgrid")
        for cvdrift0, gbdrift0, tgrid in zip(arrays.cvdrift0[0,0], arrays.gbdrift0[0,0], phi):
            f.write(f"\n{cvdrift0} {gbdrift0} {tgrid}")
        f.write("\nRplot Rprime tgrid")
        for tgrid in phi:
            f.write(f"\n0.0 0.0 {tgrid}")
        f.write("\nZplot Rprime tgrid")
        for tgrid in phi:
            f.write(f"\n0.0 0.0 {tgrid}")
        f.write("\naplot Rprime tgrid")
        for tgrid in phi:
            f.write(f"\n0.0 0.0 {tgrid}")