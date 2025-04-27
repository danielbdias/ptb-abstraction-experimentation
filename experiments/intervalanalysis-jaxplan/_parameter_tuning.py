import pyRDDLGym
from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter

from _config_tuning import experiments, tuning_seed, eval_trials, num_workers, gp_iters

if __name__ == '__main__':
    for domain_instance_experiment in experiments:
        print('--------------------------------------------------------------------------------')
        print('Parameter Tuning - ', domain_instance_experiment.domain_name, domain_instance_experiment.instance_name)
        print('--------------------------------------------------------------------------------')
        print()
        
        domain = domain_instance_experiment.domain_name
        instance = domain_instance_experiment.instance_name
        drp_template_file = domain_instance_experiment.drp_template_file
        with open(drp_template_file, 'r') as file: 
            drp_config_template = file.read() 

        # map parameters in the config that will be tuned
        def power_10(x): return 10.0 ** x
        def power_2(x): return 2.0 ** x
            
        hyperparams = [
            Hyperparameter('TUNABLE_WEIGHT', -1., 5., power_10),  # tune weight from 10^-1 ... 10^5
            Hyperparameter('TUNABLE_LEARNING_RATE', -5., 1., power_10),   # tune lr from 10^-5 ... 10^1
            Hyperparameter('TUNABLE_TOPOLOGY_FIRST_LAYER', 4, 8, power_2),  # tune weight from 16 ... 256
            Hyperparameter('TUNABLE_TOPOLOGY_SECOND_LAYER', 4, 8, power_2),   # tune lr from 16 ... 256
        ]

        # set up the environment   
        env = pyRDDLGym.make(domain, instance, vectorized=True)
    
        # build the tuner and tune
        tuning = JaxParameterTuning(env=env,
                                    config_template=drp_config_template,
                                    hyperparams=hyperparams,
                                    online=False,
                                    eval_trials=eval_trials,
                                    num_workers=num_workers,
                                    gp_iters=gp_iters)
        
        best_params = tuning.tune(key=tuning_seed, log_file=f'_intermediate/log_{domain}_{instance}.csv')
        with open(f'_results/_best_params_{domain}_{instance}.txt', 'w') as file:
            file.write(str(best_params))
