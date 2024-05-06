import os
home_directory = os.path.expanduser("~")
CONFIG = {
    10: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/quasilinear/results_March1_2024/HSX/input.HSX_QHS_vacuum_ns201',
        "output_dir": 'HSX',
        "wout": 'wout_HSX_QHS_vacuum_ns201.nc',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.0,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 8,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 9.96,
        "nfp": 4,
    },
    9: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/quasilinear/results_March1_2024/W7X/input.W7-X_standard_configuration',
        "output_dir": 'W7-X',
        "wout": 'wout_W7-X_standard_configuration.nc',
        "params": { 'nphi': 89,'nlambda': 29,'nperiod': 1.2,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 8,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 10.75,
        "nfp": 5,
    },
    8: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp4_QI',
        "output_dir": 'nfp4_QI',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 4,
    },
    7: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp3_QI',
        "output_dir": 'nfp3_QI',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 3,
    },
    6: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp2_QI',
        "output_dir": 'nfp2_QI',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 2,
    },
    5: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp3_QH',
        "output_dir": 'nfp3_QH',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 3,
    },
    4: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp3_QA',
        "output_dir": 'nfp3_QA',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 3,
    },
    3: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp1_QI',
        "output_dir": 'nfp1_QI',
        "params": { 'nphi': 101,'nlambda': 23,'nperiod': 2.0,'nstep': 300,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 4.0,'naky': 8,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 1,
    },
    2: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp4_QH',
        "output_dir": 'nfp4_QH',
        "params": { 'nphi': 121,'nlambda': 25,'nperiod': 2.5,'nstep': 350,'dt': 0.4,
                    'aky_min': 0.3,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 4,
    },
    1: {
        "input_file": f'{home_directory}/local/microstability_optimization/src/vmec_inputs/input.nfp2_QA',
        "output_dir": 'nfp2_QA',
        "params": { 'nphi': 89,'nlambda': 25,'nperiod': 3.0,'nstep': 280,'dt': 0.4,
                    'aky_min': 0.4,'aky_max': 3.0,'naky': 6,'LN': 1.0,'LT': 3.0,
                    's_radius': 0.25,'alpha_fieldline': 0,'ngauss': 3,'negrid': 8,'vnewk': 0.01
                  },
        "aspect_ratio_target": 7,
        "nfp": 2,
    }
}