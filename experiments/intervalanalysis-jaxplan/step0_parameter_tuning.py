from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter

from _config_multiprocess import num_workers
from _config_tuning import run_tuning, tuning_seed, gp_iters
from _config_run import get_experiments
from _fileio import save_raw_data, read_file
from _experiment import prepare_parallel_experiment_on_main

import os
import json

root_folder = os.path.dirname(__file__)

def power_2(x): return int(2 ** x)
def power_10(x): return 10.0 ** x
    
if __name__ == '__main__' and run_tuning:
    prepare_parallel_experiment_on_main()
    
    experiments = get_experiments()
    
    # the tuner has a para
    for domain_instance_experiment in experiments:
<<<<<<< HEAD
=======
        file_to_save = f'{root_folder}/_hyperparam_results/_best_params_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}'
        if file_exists(f'{file_to_save}.json'):
            print('--------------------------------------------------------------------------------')
            print('Skipping Tuning - ', domain_instance_experiment.domain_name, domain_instance_experiment.instance_name)
            print('--------------------------------------------------------------------------------')
            continue
        
>>>>>>> daffd77 (adding skip tuning step)
        print('--------------------------------------------------------------------------------')
        print('Parameter Tuning - ', domain_instance_experiment.domain_name, domain_instance_experiment.instance_name)
        print()
        
        drp_config_template = read_file(f"{root_folder}/{domain_instance_experiment.drp_experiment_params.tuning_params.drp_template_file}")

        print('--------------------------------------------------------------------------------')
            
        hyperparams = [
            Hyperparameter('MODEL_WEIGHT_TUNE', -1., 4., power_10),
            Hyperparameter('POLICY_WEIGHT_TUNE', -2., 2., power_10),
            Hyperparameter('LEARNING_RATE_TUNE', -5., 0., power_10),
            Hyperparameter('VARIANCE_TUNE', -2., 2., power_10),
            Hyperparameter('LAYER1_TUNE', 3, 8, power_2)
        ]

        # set up the environment   
        env = domain_instance_experiment.get_pyrddlgym_environment(root_folder, vectorized=True)

        # build the tuner and tune
        tuning = JaxParameterTuning(env=env,
                                    config_template=drp_config_template,
                                    hyperparams=hyperparams,
                                    online=False,
                                    eval_trials=domain_instance_experiment.drp_experiment_params.tuning_params.eval_trials,
                                    num_workers=num_workers,
                                    gp_iters=gp_iters)
        
        best_params = tuning.tune(key=tuning_seed, log_file=f'{root_folder}/_intermediate/log_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.csv')
        
        print('--------------------------------------------------------------------------------')
        print('Tuning Complete')
        print('Saving Best Parameters')
        print('--------------------------------------------------------------------------------')
        
        file_to_save = f'{root_folder}/_hyperparam_results/_best_params_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}'
        
        save_raw_data(str(best_params), f'{file_to_save}.txt')
        
        # convert best_params to json
        params = {
            'MODEL_WEIGHT_TUNE': float(best_params['MODEL_WEIGHT_TUNE']), 
            'POLICY_WEIGHT_TUNE': float(best_params['POLICY_WEIGHT_TUNE']),
            'LEARNING_RATE_TUNE': float(best_params['LEARNING_RATE_TUNE']),
            'VARIANCE_TUNE': float(best_params['VARIANCE_TUNE']),
            'LAYER1_TUNE': best_params['LAYER1_TUNE']
        }
        save_raw_data(json.dumps(params), f'{file_to_save}.json')
