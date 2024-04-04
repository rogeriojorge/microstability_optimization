# script-version: 2.0
# Catalyst state generated using paraview version 5.12.0-RC2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1786, 1652]
renderView1.AxesGrid = 'Grid Axes 3D Actor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [0.0426338908786813, 0.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-3.09170536994934, -0.6032679492860165, -3.909874811065417]
renderView1.CameraFocalPoint = [0.042633890878681295, 1.2622007488894682e-17, 3.667526704320342e-17]
renderView1.CameraViewUp = [0.7789409546967944, -0.204149547898699, -0.5929367176932151]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.9126047304802871
renderView1.LegendGrid = 'Legend Grid Actor'
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1786, 1652)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_17vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_17.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order3_l02/coils/fieldlines_bs_17.vtu'])
fieldlines_bs_17vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_6vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_6.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/helical_coil_stellarator/optimization_QA_asymcoils/coils/fieldlines_bs_6.vtu'])
fieldlines_bs_6vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode2vtu = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode2.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order4_length1.1_cc0.04_curvature100_msc100_mirror0.55_planar/coils/curves_after_stage2_maxmode2.vtu'])
curves_after_stage2_maxmode2vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_6vtu_1 = XMLUnstructuredGridReader(registrationName='fieldlines_bs_6.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order6_length1.3_cc0.04_curvature100_msc100_mirror0.33_nonplanar/coils/fieldlines_bs_6.vtu'])
fieldlines_bs_6vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/dual_stellarator/optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0/surf_assess_coils.vts'])
surf_assess_coilsvts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_0vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_0.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils1_nonplanar_symcoils_extracoils_verygood/coils/fieldlines_bsh_0.vtu'])
fieldlines_bsh_0vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode5vtu = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/helical_coil_stellarator/optimization_QA_asymcoils/coils/curves_after_stage2_maxmode5.vtu'])
curves_after_stage2_maxmode5vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode1vtu = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode1.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order6_length1.3_cc0.04_curvature100_msc100_mirror0.33_nonplanar/coils/curves_opt_maxmode1.vtu'])
curves_opt_maxmode1vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_big_after_stage2_maxmode2vts = XMLStructuredGridReader(registrationName='surf_big_after_stage2_maxmode2.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order4_length1.1_cc0.04_curvature100_msc100_mirror0.55_planar/coils/surf_big_after_stage2_maxmode2.vts'])
surf_big_after_stage2_maxmode2vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode3vtu = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode3.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order3_l02/coils/curves_opt_maxmode3.vtu'])
curves_opt_maxmode3vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_1 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_l01_order3_ok/surf_assess_coils.vts'])
surf_assess_coilsvts_1.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_2 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/helical_coil_stellarator/optimization_QA_asymcoils/surf_assess_coils.vts'])
surf_assess_coilsvts_2.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_3 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_nonplanar_verygood/surf_assess_coils.vts'])
surf_assess_coilsvts_3.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_1vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_1.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_nonplanar_verygood/coils/fieldlines_bsh_1.vtu'])
fieldlines_bsh_1vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
qA_final_fieldlines_bsh_10vtu = XMLUnstructuredGridReader(registrationName='QA_final_fieldlines_bsh_10.vtu', FileName=['/Users/rogeriojorge/local/dual_stellarator/optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0/coils/QA_final_fieldlines_bsh_10.vtu'])
qA_final_fieldlines_bsh_10vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
qH_final_fieldlines_bsh_14vtu = XMLUnstructuredGridReader(registrationName='QH_final_fieldlines_bsh_14.vtu', FileName=['/Users/rogeriojorge/local/dual_stellarator/optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0/coils/QH_final_fieldlines_bsh_14.vtu'])
qH_final_fieldlines_bsh_14vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode5vtu_1 = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_nonplanar_verygood/coils/curves_after_stage2_maxmode5.vtu'])
curves_after_stage2_maxmode5vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode5vtu = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils3_nonplanar_symcoils_extracoils_verygood/coils/curves_opt_maxmode5.vtu'])
curves_opt_maxmode5vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_4 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils1_nonplanar_symcoils_extracoils_verygood/surf_assess_coils.vts'])
surf_assess_coilsvts_4.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode5vtu_1 = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QH_ncoils3_nonplanar_symcoils_verygood/coils/curves_opt_maxmode5.vtu'])
curves_opt_maxmode5vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_5 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QH_ncoils3_nonplanar_symcoils_verygood/surf_assess_coils.vts'])
surf_assess_coilsvts_5.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
curves_QA_finalvtu = XMLUnstructuredGridReader(registrationName='curves_QA_final.vtu', FileName=['/Users/rogeriojorge/local/dual_stellarator/optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0/coils/curves_QA_final.vtu'])
curves_QA_finalvtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_6vtu_2 = XMLUnstructuredGridReader(registrationName='fieldlines_bs_6.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order0_length1.0_cc0.04_curvature50_msc50_mirror0.55_planar/coils/fieldlines_bs_6.vtu'])
fieldlines_bs_6vtu_2.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode4vtu = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode4.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.209_Aminor0.0982/coils/curves_after_stage2_maxmode4.vtu'])
curves_after_stage2_maxmode4vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_big_after_stage2_maxmode4vts = XMLStructuredGridReader(registrationName='surf_big_after_stage2_maxmode4.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.209_Aminor0.0982/coils/surf_big_after_stage2_maxmode4.vts'])
surf_big_after_stage2_maxmode4vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_5vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order4_length1.1_cc0.04_curvature100_msc100_mirror0.55_planar/coils/fieldlines_bs_5.vtu'])
fieldlines_bs_5vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode2vtu = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode2.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order0_length1.0_cc0.04_curvature50_msc50_mirror0.55_planar/coils/curves_opt_maxmode2.vtu'])
curves_opt_maxmode2vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_19vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_19.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.21_Aminor0.0982/coils/fieldlines_bs_19.vtu'])
fieldlines_bs_19vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode5vtu_2 = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils1_nonplanar_symcoils_extracoils_verygood/coils/curves_opt_maxmode5.vtu'])
curves_opt_maxmode5vtu_2.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_big_opt_maxmode3vts = XMLStructuredGridReader(registrationName='surf_big_opt_maxmode3.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order3_l02/coils/surf_big_opt_maxmode3.vts'])
surf_big_opt_maxmode3vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_big_opt_maxmode2vts = XMLStructuredGridReader(registrationName='surf_big_opt_maxmode2.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order0_length1.0_cc0.04_curvature50_msc50_mirror0.55_planar/coils/surf_big_opt_maxmode2.vts'])
surf_big_opt_maxmode2vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode5vtu_3 = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils2_nonplanar_symcoils_extracoils_extragood/coils/curves_opt_maxmode5.vtu'])
curves_opt_maxmode5vtu_3.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_10vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_10.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order4_length1.1_cc0.04_curvature100_msc100_mirror0.55_planar/coils/fieldlines_bs_10.vtu'])
fieldlines_bs_10vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_big_opt_maxmode1vts = XMLStructuredGridReader(registrationName='surf_big_opt_maxmode1.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order6_length1.3_cc0.04_curvature100_msc100_mirror0.33_nonplanar/coils/surf_big_opt_maxmode1.vts'])
surf_big_opt_maxmode1vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode5vtu_2 = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.21_Aminor0.0982/coils/curves_after_stage2_maxmode5.vtu'])
curves_after_stage2_maxmode5vtu_2.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_18vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_18.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.209_Aminor0.0982/coils/fieldlines_bs_18.vtu'])
fieldlines_bs_18vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_big_after_stage2_maxmode5vts = XMLStructuredGridReader(registrationName='surf_big_after_stage2_maxmode5.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_order4_l02_Rmajor0.21_Aminor0.0982/coils/surf_big_after_stage2_maxmode5.vts'])
surf_big_after_stage2_maxmode5vts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_6 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_planar_ok/surf_assess_coils.vts'])
surf_assess_coilsvts_6.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_7 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils2_nonplanar_symcoils_extracoils_extragood/surf_assess_coils.vts'])
surf_assess_coilsvts_7.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Structured Grid Reader'
surf_1_opt_bigvts = XMLStructuredGridReader(registrationName='surf_1_opt_big.vts', FileName=['/Users/rogeriojorge/local/dual_stellarator/optimization_good_ncoils_7_order_5_R1_0.47_length_target_2.2_weight_10000.0_max_curvature_6.5_weight_1.4e-05_msc_14.0_weight_3.1e-05_cc_0.0758_weight_1100.0_QH_0.03_weight_20000.0/coils/surf_1_opt_big.vts'])
surf_1_opt_bigvts.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_9vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_9.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_planar_ok/coils/fieldlines_bsh_9.vtu'])
fieldlines_bsh_9vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_opt_maxmode3vtu_1 = XMLUnstructuredGridReader(registrationName='curves_opt_maxmode3.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_simple_nfp4_ncoils1_planar_ok/coils/curves_opt_maxmode3.vtu'])
curves_opt_maxmode3vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_6vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_6.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QH_ncoils3_nonplanar_symcoils_verygood/coils/fieldlines_bsh_6.vtu'])
fieldlines_bsh_6vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_10vtu_1 = XMLUnstructuredGridReader(registrationName='fieldlines_bs_10.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order6_length1.3_cc0.04_curvature100_msc100_mirror0.33_nonplanar/coils/fieldlines_bs_10.vtu'])
fieldlines_bs_10vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_3vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_3.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils3_nonplanar_symcoils_extracoils_verygood/coils/fieldlines_bsh_3.vtu'])
fieldlines_bsh_3vtu.PointArrayStatus = ['idx']

# create a new 'XML Structured Grid Reader'
surf_assess_coilsvts_8 = XMLStructuredGridReader(registrationName='surf_assess_coils.vts', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils3_nonplanar_symcoils_extracoils_verygood/surf_assess_coils.vts'])
surf_assess_coilsvts_8.PointArrayStatus = ['dphi x dtheta', 'dphi', 'dtheta', 'B.n/B']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bsh_1vtu_1 = XMLUnstructuredGridReader(registrationName='fieldlines_bsh_1.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/circular_coil_stellarator/optimization_QA_ncoils2_nonplanar_symcoils_extracoils_extragood/coils/fieldlines_bsh_1.vtu'])
fieldlines_bsh_1vtu_1.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_15vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_15.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_l01_order3_ok/coils/fieldlines_bs_15.vtu'])
fieldlines_bs_15vtu.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
curves_after_stage2_maxmode5vtu_3 = XMLUnstructuredGridReader(registrationName='curves_after_stage2_maxmode5.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/etos_helical_coil/optimization_QA_asymcoils_l01_order3_ok/coils/curves_after_stage2_maxmode5.vtu'])
curves_after_stage2_maxmode5vtu_3.PointArrayStatus = ['idx']

# create a new 'XML Unstructured Grid Reader'
fieldlines_bs_11vtu = XMLUnstructuredGridReader(registrationName='fieldlines_bs_11.vtu', FileName=['/Users/rogeriojorge/local/microstability_optimization/src/joaquim_circular_coil/optimization_simple_nfp3_order0_length1.0_cc0.04_curvature50_msc50_mirror0.55_planar/coils/fieldlines_bs_11.vtu'])
fieldlines_bs_11vtu.PointArrayStatus = ['idx']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from surf_1_opt_bigvts
surf_1_opt_bigvtsDisplay = Show(surf_1_opt_bigvts, renderView1, 'StructuredGridRepresentation')

# get 2D transfer function for 'BnB'
bnBTF2D = GetTransferFunction2D('BnB')
bnBTF2D.ScalarRangeInitialized = 1
bnBTF2D.Range = [-0.03, 0.03, 0.0, 1.0]

# get color transfer function/color map for 'BnB'
bnBLUT = GetColorTransferFunction('BnB')
bnBLUT.AutomaticRescaleRangeMode = 'Never'
bnBLUT.TransferFunction2D = bnBTF2D
bnBLUT.RGBPoints = [-0.03, 0.231373, 0.298039, 0.752941, 0.0, 0.865003, 0.865003, 0.865003, 0.03, 0.705882, 0.0156863, 0.14902]
bnBLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'BnB'
bnBPWF = GetOpacityTransferFunction('BnB')
bnBPWF.Points = [-0.03, 0.0, 0.5, 0.0, 0.03, 1.0, 0.5, 0.0]
bnBPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
surf_1_opt_bigvtsDisplay.Representation = 'Surface'
surf_1_opt_bigvtsDisplay.ColorArrayName = ['POINTS', 'B.n/B']
surf_1_opt_bigvtsDisplay.LookupTable = bnBLUT
surf_1_opt_bigvtsDisplay.SelectTCoordArray = 'None'
surf_1_opt_bigvtsDisplay.SelectNormalArray = 'None'
surf_1_opt_bigvtsDisplay.SelectTangentArray = 'None'
surf_1_opt_bigvtsDisplay.OSPRayScaleArray = 'B.n/B'
surf_1_opt_bigvtsDisplay.OSPRayScaleFunction = 'Piecewise Function'
surf_1_opt_bigvtsDisplay.Assembly = ''
surf_1_opt_bigvtsDisplay.SelectOrientationVectors = 'dphi x dtheta'
surf_1_opt_bigvtsDisplay.ScaleFactor = 0.20923037908034853
surf_1_opt_bigvtsDisplay.SelectScaleArray = 'B.n/B'
surf_1_opt_bigvtsDisplay.GlyphType = 'Arrow'
surf_1_opt_bigvtsDisplay.GlyphTableIndexArray = 'B.n/B'
surf_1_opt_bigvtsDisplay.GaussianRadius = 0.010461518954017425
surf_1_opt_bigvtsDisplay.SetScaleArray = ['POINTS', 'B.n/B']
surf_1_opt_bigvtsDisplay.ScaleTransferFunction = 'Piecewise Function'
surf_1_opt_bigvtsDisplay.OpacityArray = ['POINTS', 'B.n/B']
surf_1_opt_bigvtsDisplay.OpacityTransferFunction = 'Piecewise Function'
surf_1_opt_bigvtsDisplay.DataAxesGrid = 'Grid Axes Representation'
surf_1_opt_bigvtsDisplay.PolarAxes = 'Polar Axes Representation'
surf_1_opt_bigvtsDisplay.ScalarOpacityFunction = bnBPWF
surf_1_opt_bigvtsDisplay.ScalarOpacityUnitDistance = 0.16912172616353136
surf_1_opt_bigvtsDisplay.SelectInputVectors = ['POINTS', 'dphi x dtheta']
surf_1_opt_bigvtsDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
surf_1_opt_bigvtsDisplay.ScaleTransferFunction.Points = [-0.01618174522842306, 0.0, 0.5, 0.0, 0.016181745228423302, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
surf_1_opt_bigvtsDisplay.OpacityTransferFunction.Points = [-0.01618174522842306, 0.0, 0.5, 0.0, 0.016181745228423302, 1.0, 0.5, 0.0]

# show data from curves_QA_finalvtu
curves_QA_finalvtuDisplay = Show(curves_QA_finalvtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
curves_QA_finalvtuDisplay.Representation = 'Surface'
curves_QA_finalvtuDisplay.ColorArrayName = ['POINTS', '']
curves_QA_finalvtuDisplay.LineWidth = 11.0
curves_QA_finalvtuDisplay.RenderLinesAsTubes = 1
curves_QA_finalvtuDisplay.SelectTCoordArray = 'None'
curves_QA_finalvtuDisplay.SelectNormalArray = 'None'
curves_QA_finalvtuDisplay.SelectTangentArray = 'None'
curves_QA_finalvtuDisplay.OSPRayScaleArray = 'idx'
curves_QA_finalvtuDisplay.OSPRayScaleFunction = 'Piecewise Function'
curves_QA_finalvtuDisplay.Assembly = ''
curves_QA_finalvtuDisplay.SelectOrientationVectors = 'None'
curves_QA_finalvtuDisplay.ScaleFactor = 0.2669483859249322
curves_QA_finalvtuDisplay.SelectScaleArray = 'idx'
curves_QA_finalvtuDisplay.GlyphType = 'Arrow'
curves_QA_finalvtuDisplay.GlyphTableIndexArray = 'idx'
curves_QA_finalvtuDisplay.GaussianRadius = 0.01334741929624661
curves_QA_finalvtuDisplay.SetScaleArray = ['POINTS', 'idx']
curves_QA_finalvtuDisplay.ScaleTransferFunction = 'Piecewise Function'
curves_QA_finalvtuDisplay.OpacityArray = ['POINTS', 'idx']
curves_QA_finalvtuDisplay.OpacityTransferFunction = 'Piecewise Function'
curves_QA_finalvtuDisplay.DataAxesGrid = 'Grid Axes Representation'
curves_QA_finalvtuDisplay.PolarAxes = 'Polar Axes Representation'
curves_QA_finalvtuDisplay.ScalarOpacityUnitDistance = 1.1004545855791423
curves_QA_finalvtuDisplay.OpacityArrayName = ['POINTS', 'idx']
curves_QA_finalvtuDisplay.SelectInputVectors = [None, '']
curves_QA_finalvtuDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
curves_QA_finalvtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 41.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
curves_QA_finalvtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 41.0, 1.0, 0.5, 0.0]

# show data from surf_assess_coilsvts
surf_assess_coilsvtsDisplay = Show(surf_assess_coilsvts, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
surf_assess_coilsvtsDisplay.Representation = 'Surface'
surf_assess_coilsvtsDisplay.ColorArrayName = ['POINTS', 'B.n/B']
surf_assess_coilsvtsDisplay.LookupTable = bnBLUT
surf_assess_coilsvtsDisplay.SelectTCoordArray = 'None'
surf_assess_coilsvtsDisplay.SelectNormalArray = 'None'
surf_assess_coilsvtsDisplay.SelectTangentArray = 'None'
surf_assess_coilsvtsDisplay.OSPRayScaleArray = 'B.n/B'
surf_assess_coilsvtsDisplay.OSPRayScaleFunction = 'Piecewise Function'
surf_assess_coilsvtsDisplay.Assembly = ''
surf_assess_coilsvtsDisplay.SelectOrientationVectors = 'dphi x dtheta'
surf_assess_coilsvtsDisplay.ScaleFactor = 0.19151100350377448
surf_assess_coilsvtsDisplay.SelectScaleArray = 'B.n/B'
surf_assess_coilsvtsDisplay.GlyphType = 'Arrow'
surf_assess_coilsvtsDisplay.GlyphTableIndexArray = 'B.n/B'
surf_assess_coilsvtsDisplay.GaussianRadius = 0.009575550175188723
surf_assess_coilsvtsDisplay.SetScaleArray = ['POINTS', 'B.n/B']
surf_assess_coilsvtsDisplay.ScaleTransferFunction = 'Piecewise Function'
surf_assess_coilsvtsDisplay.OpacityArray = ['POINTS', 'B.n/B']
surf_assess_coilsvtsDisplay.OpacityTransferFunction = 'Piecewise Function'
surf_assess_coilsvtsDisplay.DataAxesGrid = 'Grid Axes Representation'
surf_assess_coilsvtsDisplay.PolarAxes = 'Polar Axes Representation'
surf_assess_coilsvtsDisplay.ScalarOpacityFunction = bnBPWF
surf_assess_coilsvtsDisplay.ScalarOpacityUnitDistance = 0.11523763118993562
surf_assess_coilsvtsDisplay.SelectInputVectors = ['POINTS', 'dphi x dtheta']
surf_assess_coilsvtsDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
surf_assess_coilsvtsDisplay.ScaleTransferFunction.Points = [-0.06495848145701023, 0.0, 0.5, 0.0, 0.06495848145701082, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
surf_assess_coilsvtsDisplay.OpacityTransferFunction.Points = [-0.06495848145701023, 0.0, 0.5, 0.0, 0.06495848145701082, 1.0, 0.5, 0.0]

# show data from qH_final_fieldlines_bsh_14vtu
qH_final_fieldlines_bsh_14vtuDisplay = Show(qH_final_fieldlines_bsh_14vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
qH_final_fieldlines_bsh_14vtuDisplay.Representation = 'Surface'
qH_final_fieldlines_bsh_14vtuDisplay.ColorArrayName = ['POINTS', '']
qH_final_fieldlines_bsh_14vtuDisplay.DiffuseColor = [0.0, 0.0, 0.0]
qH_final_fieldlines_bsh_14vtuDisplay.Opacity = 0.23
qH_final_fieldlines_bsh_14vtuDisplay.SelectTCoordArray = 'None'
qH_final_fieldlines_bsh_14vtuDisplay.SelectNormalArray = 'None'
qH_final_fieldlines_bsh_14vtuDisplay.SelectTangentArray = 'None'
qH_final_fieldlines_bsh_14vtuDisplay.OSPRayScaleArray = 'idx'
qH_final_fieldlines_bsh_14vtuDisplay.OSPRayScaleFunction = 'Piecewise Function'
qH_final_fieldlines_bsh_14vtuDisplay.Assembly = ''
qH_final_fieldlines_bsh_14vtuDisplay.SelectOrientationVectors = 'None'
qH_final_fieldlines_bsh_14vtuDisplay.ScaleFactor = 0.19311397414334275
qH_final_fieldlines_bsh_14vtuDisplay.SelectScaleArray = 'idx'
qH_final_fieldlines_bsh_14vtuDisplay.GlyphType = 'Arrow'
qH_final_fieldlines_bsh_14vtuDisplay.GlyphTableIndexArray = 'idx'
qH_final_fieldlines_bsh_14vtuDisplay.GaussianRadius = 0.009655698707167138
qH_final_fieldlines_bsh_14vtuDisplay.SetScaleArray = ['POINTS', 'idx']
qH_final_fieldlines_bsh_14vtuDisplay.ScaleTransferFunction = 'Piecewise Function'
qH_final_fieldlines_bsh_14vtuDisplay.OpacityArray = ['POINTS', 'idx']
qH_final_fieldlines_bsh_14vtuDisplay.OpacityTransferFunction = 'Piecewise Function'
qH_final_fieldlines_bsh_14vtuDisplay.DataAxesGrid = 'Grid Axes Representation'
qH_final_fieldlines_bsh_14vtuDisplay.PolarAxes = 'Polar Axes Representation'
qH_final_fieldlines_bsh_14vtuDisplay.ScalarOpacityUnitDistance = 2.70761245089914
qH_final_fieldlines_bsh_14vtuDisplay.OpacityArrayName = ['POINTS', 'idx']
qH_final_fieldlines_bsh_14vtuDisplay.SelectInputVectors = [None, '']
qH_final_fieldlines_bsh_14vtuDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
qH_final_fieldlines_bsh_14vtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
qH_final_fieldlines_bsh_14vtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# show data from qA_final_fieldlines_bsh_10vtu
qA_final_fieldlines_bsh_10vtuDisplay = Show(qA_final_fieldlines_bsh_10vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
qA_final_fieldlines_bsh_10vtuDisplay.Representation = 'Surface'
qA_final_fieldlines_bsh_10vtuDisplay.ColorArrayName = ['POINTS', '']
qA_final_fieldlines_bsh_10vtuDisplay.DiffuseColor = [0.0, 0.0, 0.0]
qA_final_fieldlines_bsh_10vtuDisplay.Opacity = 0.2
qA_final_fieldlines_bsh_10vtuDisplay.SelectTCoordArray = 'None'
qA_final_fieldlines_bsh_10vtuDisplay.SelectNormalArray = 'None'
qA_final_fieldlines_bsh_10vtuDisplay.SelectTangentArray = 'None'
qA_final_fieldlines_bsh_10vtuDisplay.OSPRayScaleArray = 'idx'
qA_final_fieldlines_bsh_10vtuDisplay.OSPRayScaleFunction = 'Piecewise Function'
qA_final_fieldlines_bsh_10vtuDisplay.Assembly = ''
qA_final_fieldlines_bsh_10vtuDisplay.SelectOrientationVectors = 'None'
qA_final_fieldlines_bsh_10vtuDisplay.ScaleFactor = 0.21710584693280455
qA_final_fieldlines_bsh_10vtuDisplay.SelectScaleArray = 'idx'
qA_final_fieldlines_bsh_10vtuDisplay.GlyphType = 'Arrow'
qA_final_fieldlines_bsh_10vtuDisplay.GlyphTableIndexArray = 'idx'
qA_final_fieldlines_bsh_10vtuDisplay.GaussianRadius = 0.010855292346640226
qA_final_fieldlines_bsh_10vtuDisplay.SetScaleArray = ['POINTS', 'idx']
qA_final_fieldlines_bsh_10vtuDisplay.ScaleTransferFunction = 'Piecewise Function'
qA_final_fieldlines_bsh_10vtuDisplay.OpacityArray = ['POINTS', 'idx']
qA_final_fieldlines_bsh_10vtuDisplay.OpacityTransferFunction = 'Piecewise Function'
qA_final_fieldlines_bsh_10vtuDisplay.DataAxesGrid = 'Grid Axes Representation'
qA_final_fieldlines_bsh_10vtuDisplay.PolarAxes = 'Polar Axes Representation'
qA_final_fieldlines_bsh_10vtuDisplay.ScalarOpacityUnitDistance = 3.0857854248704415
qA_final_fieldlines_bsh_10vtuDisplay.OpacityArrayName = ['POINTS', 'idx']
qA_final_fieldlines_bsh_10vtuDisplay.SelectInputVectors = [None, '']
qA_final_fieldlines_bsh_10vtuDisplay.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
qA_final_fieldlines_bsh_10vtuDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
qA_final_fieldlines_bsh_10vtuDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for bnBLUT in view renderView1
bnBLUTColorBar = GetScalarBar(bnBLUT, renderView1)
bnBLUTColorBar.Orientation = 'Horizontal'
bnBLUTColorBar.WindowLocation = 'Any Location'
bnBLUTColorBar.Position = [0.14525064225018106, 0.8894915254237291]
bnBLUTColorBar.Title = 'B.n/B'
bnBLUTColorBar.ComponentTitle = ''
bnBLUTColorBar.ScalarBarLength = 0.6567495807005723

# set color bar visibility
bnBLUTColorBar.Visibility = 1

# show color legend
surf_1_opt_bigvtsDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
surf_assess_coilsvtsDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = renderView1
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.0

# initialize the animation scene

# ----------------------------------------------------------------
# restore active source
SetActiveSource(surf_assess_coilsvts_1)
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'Time Step'
options.CatalystLiveTrigger = 'Time Step'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
