import os
import time
import jax

from multiprocessing import get_context, freeze_support

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxDeepReactivePolicy

from _domains import domains, jax_seeds, silent
from _utils import run_experiment, save_data

root_folder = os.path.dirname(__file__)

def perform_experiment(domain):
    if domain.experiment_params.is_drp():
        return perform_drp_experiment(domain)
    
    return perform_slp_experiment(domain)

def perform_drp_experiment(domain):
    print(f'[{os.getpid()}] Domain: ', domain.name, ' Instance: ', domain.instance, ' DRP')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain.name} (regular) - DRP"

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxDeepReactivePolicy(domain.experiment_params.topology)
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        env_params = experiment_params

        experiment_summary = run_experiment(regular_experiment_name, rddl_model=grounded_model, planner_parameters=env_params, silent=silent)
        regular_env_experiment_stats.append(experiment_summary)

    save_data(regular_env_experiment_stats, f'{root_folder}/_results/baseline_drp_run_data_{domain.name}_{domain.instance}.pickle')

def perform_slp_experiment(domain):
    print(f'[{os.getpid()}] Domain: ', domain.name, ' Instance: ', domain.instance, ' SLP')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain.name} (regular) - Straight line"

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxStraightLinePlan()
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        env_params = experiment_params

        experiment_summary = run_experiment(regular_experiment_name, rddl_model=grounded_model, planner_parameters=env_params, silent=silent)
        regular_env_experiment_stats.append(experiment_summary)

    save_data(regular_env_experiment_stats, f'{root_folder}/_results/baseline_slp_run_data_{domain.name}_{domain.instance}.pickle')

if __name__ == '__main__':
    freeze_support()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create baseline Run with random policy')
    print('--------------------------------------------------------------------------------')
    print()

    start_time = time.time()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    pool_context = 'spawn'
    num_workers = 4
    timeout = 3_600 # 1 hour

    # create worker pool: note each iteration must wait for all workers
    # to finish before moving to the next
    with get_context(pool_context).Pool(processes=num_workers) as pool:
        multiple_results = [pool.apply_async(perform_experiment, args=(domain,)) for domain in domains]
        
        # wait for all workers to finish
        for res in multiple_results:
            res.get(timeout=timeout)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()