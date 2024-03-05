#!/usr/bin/env python3
import os
import shutil
import subprocess

QA_or_QH = "QH"
beta = 2.5
finite_beta_folder = "/Users/rogeriojorge/local/microstability_optimization/src/util/finite_beta"
files_to_copy = ['input.namelist', 'job.sfincsScan', 'profiles']

# prefix_save = 'optimization'
# results_folder = 'results'
# OUT_DIR_APPENDIX=f"{prefix_save}_{QA_or_QH}"
# OUT_DIR_APPENDIX+=f'_beta{beta:.1f}'
# output_path_parameters=f"{OUT_DIR_APPENDIX}.csv"
# this_path = os.path.dirname(os.path.abspath(__file__))
# OUT_DIR = os.path.join(this_path,results_folder,QA_or_QH,OUT_DIR_APPENDIX)
OUT_DIR = "/Users/rogeriojorge/local/microstability_optimization/src/finite_beta/zenodo_Matt"
os.chdir(OUT_DIR)

# Copy necessary files to the output directory
for filename in files_to_copy:
    source_path = os.path.join(finite_beta_folder, filename)
    destination_path = os.path.join(OUT_DIR, filename)
    try:
        shutil.copy2(source_path, destination_path)
        print(f"Successfully copied {filename} from {finite_beta_folder} to {OUT_DIR}")
    except FileNotFoundError:
        print(f"File {filename} not found in {finite_beta_folder}")
    except Exception as e:
        print(f"Error copying {filename} from {finite_beta_folder} to {OUT_DIR}: {e}")

def change_equilibrium_file(input_file_path, new_path):
    backup_path = input_file_path + ".bak"
    os.rename(input_file_path, backup_path)
    with open(backup_path, 'r') as backup, open(input_file_path, 'w') as original:
        original.writelines(line.replace(f'"{line.split()[1]}"', f'"{new_path}"') if 'equilibriumFile' in line else line for line in backup)
    os.remove(backup_path)
# Specify the path to your input.namelist file
input_file_path = os.path.join(OUT_DIR, "input.namelist")
# Specify the new path for equilibriumFile
new_equilibrium_path = os.path.join(OUT_DIR, "wout_final.nc")
# Call the function to change the equilibriumFile path
change_equilibrium_file(input_file_path, new_equilibrium_path)

# Run sfincsScan, sfincsScanPlot_4, and convertSfincsToVmecCurrentProfile
for script_name in ["sfincsScan", "sfincsScanPlot_4", "convertSfincsToVmecCurrentProfile"]:
    script_path = os.path.join(finite_beta_folder, script_name)
    command = ["python", script_path]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully ran {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
