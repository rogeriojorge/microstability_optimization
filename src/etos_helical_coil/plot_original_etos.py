#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simsopt.geo import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import load_coils_from_makegrid_file, Coil
this_path = str(Path(__file__).parent.resolve())
os.chdir(this_path)

order = 25
ppp = 10

loaded_coils = load_coils_from_makegrid_file("coils.etos_231", order, ppp)

gammas = [coil.curve.gamma() for coil in loaded_coils]
currents = [coil.current for coil in loaded_coils]
curves = [coil.curve for coil in loaded_coils]

curves_to_vtk(curves, "curves_etos")

# coils = [Coil(curve, current) for curve, current in zip(curves, currents)]

# bs = BiotSavart(coils)
# loaded_bs = BiotSavart(loaded_coils)

# points = np.asarray(17 * [[0.9, 0.4, -0.85]])
# points += 0.01 * (np.random.rand(*points.shape) - 0.5)
# bs.set_points(points)
# loaded_bs.set_points(points)

# B = bs.B()
# loaded_B = loaded_bs.B()

# np.testing.assert_allclose(B, loaded_B)
# print(B)