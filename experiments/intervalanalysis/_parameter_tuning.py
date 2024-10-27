import os
import sys

import jax

import pyRDDLGym

from pyRDDLGym_jax.core.tuning import JaxParameterTuningSLP
from pyRDDLGym_jax.core.planner import load_config

from _domains import domains, jax_seeds

root_folder = os.path.dirname(__file__)

def main(domain, domain_name, instance, planner_args, plan_args, train_args, trials=5, iters=20, workers=4):
    env = pyRDDLGym.make(domain, instance, vectorized=True)
    
    tuning = JaxParameterTuningSLP(env=env,
                          train_epochs=train_args['epochs'],
                          timeout_training=train_args['train_seconds'],
                          eval_trials=trials,
                          planner_kwargs=planner_args,
                          plan_kwargs=plan_args,
                          num_workers=workers,
                          gp_iters=iters)
    
    # perform tuning and report best parameters
    best = tuning.tune(key=train_args['key'], filename=f'gp_{domain_name}', save_plot=True)
    print(f'best parameters found: {best}')

if __name__ == "__main__":
    for domain in domains:
        print('--------------------------------------------------------------------------------')
        print('Domain: ', domain)
        print('--------------------------------------------------------------------------------')
        print()

        #########################################################################################################
        # Runs with regular domain (just to use as comparison)
        #########################################################################################################

        domain_path = f"{root_folder}/domains/{domain.name}"
        domain_file_path = f'{domain_path}/domain.rddl'
        instance_file_path = f'{domain_path}/{domain.instance}.rddl'
        
        train_args = {
            'key': jax.random.PRNGKey(jax_seeds[0]),
            'epochs': domain.experiment_params.training_params.epochs,
            'train_seconds': domain.experiment_params.training_params.train_seconds
        }
        
        planner_args = { # Optimizer
            'batch_size_train': domain.experiment_params.optimizer_params.batch_size_train,
            'batch_size_test': domain.experiment_params.optimizer_params.batch_size_test,
            'optimizer': domain.experiment_params.optimizer_params.optimizer,
        }
        
        plan_args = {} # Model
        
        main(domain_file_path, domain.name, instance_file_path, planner_args, plan_args, train_args)