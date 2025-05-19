from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter

from _config_tuning import experiments, tuning_seed, eval_trials, num_workers, gp_iters
from _fileio import save_raw_data, read_file
from _experiment import prepare_parallel_experiment_on_main

import os

root_folder = os.path.dirname(__file__)

def power_2(x): return int(2 ** x)
def power_10(x): return 10.0 ** x
    
if __name__ == '__main__':
    prepare_parallel_experiment_on_main()
    
    # the tuner has a para
    for domain_instance_experiment in experiments:
        print('--------------------------------------------------------------------------------')
        print('Parameter Tuning - ', domain_instance_experiment.domain_name, domain_instance_experiment.instance_name)
        print()
        
        # domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)
        domain_file_path = domain_instance_experiment.domain_name
        instance_file_path = domain_instance_experiment.instance_name
        
        drp_config_template = read_file(f"{root_folder}/{domain_instance_experiment.drp_template_file}")

        print('Domain file: ', domain_file_path)
        print('Instance file: ', instance_file_path)
        print()
        print('--------------------------------------------------------------------------------')
            
        hyperparams = [
            Hyperparameter('MODEL_WEIGHT_TUNE', -1., 4., power_10),
            Hyperparameter('POLICY_WEIGHT_TUNE', -2., 2., power_10),
            Hyperparameter('LEARNING_RATE_TUNE', -5., 0., power_10),
            Hyperparameter('LAYER1_TUNE', 3, 8, power_2),
            Hyperparameter('LAYER2_TUNE', 3, 8, power_2)
        ]

        # set up the environment   
        env = domain_instance_experiment.get_pyrddlgym_environment(root_folder, vectorized=True)

        # build the tuner and tune
        tuning = JaxParameterTuning(env=env,
                                    config_template=drp_config_template,
                                    hyperparams=hyperparams,
                                    online=False,
                                    eval_trials=eval_trials,
                                    num_workers=num_workers,
                                    gp_iters=gp_iters)
        
        best_params = tuning.tune(key=tuning_seed, log_file=f'{root_folder}/_intermediate/log_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.csv')
        save_raw_data(str(best_params), f'{root_folder}/_results/_best_params_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.txt')
