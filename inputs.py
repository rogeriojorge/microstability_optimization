## inputs.py
# Description: Input parameters for the microstability optimization
import numpy as np
code_to_use = 'gs2' # 'gs2', 'GX' or 'stella'
linear_or_nonlinear = 'linear' # 'linear' or 'nonlinear'
QA_or_QH_or_QI = 'QA' # 'QA', 'QH' or 'QI'
opt_turbulence = True # True or False
max_modes = [1,2,3] # List of ntor, mpol modes to be considered
########### Location of GK executables ###########
gs2_executable = '/Users/rogeriojorge/local/gs2/bin/gs2' # Path to GS2 executable
# gs2_executable = '/marconi/home/userexternal/rjorge00/gs2/bin/gs2' # Path to GS2 executable
gx_executable = '/m100/home/userexternal/rjorge00/gx_latest/gx' # Path to GX executable
convert_VMEC_to_GX = '/m100/home/userexternal/rjorge00/gx_latest/geometry_modules/vmec/convert_VMEC_to_GX' # Path to VMEC to GX converter
stella_executable = '/Users/rogeriojorge/local/stella/build/stella' # Path to STELLA executable
# stella_executable = '/marconi/home/userexternal/rjorge00/stella/build/stella' # Path to STELLA executable
########### Optimization parameters ##############
MAXITER = 10 # Maximum number of optimization iterations
diff_method = 'centered' # 'centered' or 'forward'
local_optimization_method = 'lm' # 'trf' or 'lm'
perform_extra_solve = True # True or False
ftol=1e-6 # Relative error desired in the sum of squares.
GROWTHRATE_THRESHOLD = 10 # Growth rate if objective function evaluation fails
########### Physics input parameters #############
s_radius = 0.25
alpha_fieldline = 0
LN = 1.0
LT = 3.0
weight_aspect_ratio = 3e+0
weight_optTurbulence = 30
aspect_ratio_QA = 6
aspect_ratio_QH = 8
aspect_ratio_QI = 9
nfp_QA = 2
nfp_QH = 4
nfp_QI = 1
########### GS2 input parameters #################
nphi= 151
nlambda = 35
nperiod = 5.0
nstep = 340
dt = 0.4
aky_min = 0.3
aky_max = 3.0
naky = 10
ngauss = 3
negrid = 10
phi_GS2 = np.linspace(-nperiod*np.pi, nperiod*np.pi, nphi)
########### Extra input parameters ###############
results_folder = 'results'