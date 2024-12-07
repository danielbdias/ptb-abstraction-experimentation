import os
import time
import jax

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxDeepReactivePolicy

from multiprocessing import get_context, freeze_support

from _domains import domains, jax_seeds, silent
from _utils import run_experiment, save_data, load_data, PlannerParameters

root_folder = os.path.dirname(__file__)

def perform_experiment(domain):
    if domain.experiment_params.is_drp():
        return perform_drp_experiment(domain)
    
    return perform_slp_experiment(domain)

def perform_drp_experiment(domain):
    print(f'[{os.getpid()}] Domain: ', domain.name, ' Instance: ', domain.instance, ' DRP')

    #########################################################################################################
    # Runs PtB with modified domain (that has ground fluents frozen with initial state values)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    
    regular_domain_file_path = f'{domain_path}/domain.rddl'
    regular_instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=regular_domain_file_path, instance=regular_instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    regular_grounded_model = grounder.ground()
    
    ablated_model_file_path = f'{root_folder}/_intermediate/domain_{domain.name}_{domain.instance}.model'
    ablated_grounded_model = load_data(ablated_model_file_path)

    warm_start_creation_experiment_stats = []
    warm_start_run_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxDeepReactivePolicy(domain.experiment_params.topology)
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        env_params = experiment_params

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (warm-start creation) - DRP", rddl_model=ablated_grounded_model, planner_parameters=env_params, silent=silent)
        warm_start_creation_experiment_stats.append(abstraction_env_experiment_summary)

        initializers_per_layer = {} # TODO: improve this
        for layer_name in abstraction_env_experiment_summary.final_policy_weights.keys():
            initializers_per_layer[layer_name] = abstraction_env_experiment_summary.final_policy_weights[layer_name]

        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxDeepReactivePolicy(domain.experiment_params.topology, initializer_per_layer=initializers_per_layer)        
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        warm_start_env_params = experiment_params

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm-start initialization) - DRP", rddl_model=regular_grounded_model, planner_parameters=warm_start_env_params, silent=silent)
        warm_start_run_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(warm_start_creation_experiment_stats, f'{root_folder}/_results/warmstart_creation_drp_run_data_{domain.name}_{domain.instance}.pickle')
    save_data(warm_start_run_experiment_stats, f'{root_folder}/_results/warmstart_execution_drp_run_data_{domain.name}_{domain.instance}.pickle')

def perform_slp_experiment(domain):
    print(f'[{os.getpid()}] Domain: ', domain.name, ' Instance: ', domain.instance, ' SLP')

    #########################################################################################################
    # Runs PtB with modified domain (that has ground fluents frozen with initial state values)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    
    regular_domain_file_path = f'{domain_path}/domain.rddl'
    regular_instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=regular_domain_file_path, instance=regular_instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    regular_grounded_model = grounder.ground()
    
    ablated_model_file_path = f'{root_folder}/_intermediate/domain_{domain.name}_{domain.instance}.model'
    ablated_grounded_model = load_data(ablated_model_file_path)

    warm_start_creation_experiment_stats = []
    warm_start_run_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxStraightLinePlan()
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        env_params = experiment_params

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (warm-start creation) - Straight line", rddl_model=ablated_grounded_model, planner_parameters=env_params, silent=silent)
        warm_start_creation_experiment_stats.append(abstraction_env_experiment_summary)

        experiment_params = domain.experiment_params
        experiment_params.optimizer_params.plan = JaxStraightLinePlan()
        experiment_params.optimizer_params.guess = abstraction_env_experiment_summary.final_policy_weights
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        warm_start_env_params = experiment_params

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm-start initialization) - Straight line", rddl_model=regular_grounded_model, planner_parameters=warm_start_env_params, silent=silent)
        warm_start_run_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(warm_start_creation_experiment_stats, f'{root_folder}/_results/warmstart_creation_slp_run_data_{domain.name}_{domain.instance}.pickle')
    save_data(warm_start_run_experiment_stats, f'{root_folder}/_results/warmstart_execution_slp_run_data_{domain.name}_{domain.instance}.pickle')

start_time = time.time()

if __name__ == '__main__':
    freeze_support()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create Warm Start policies and Run with Warm Start')
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    pool_context = 'spawn'
    num_workers = 4
    timeout = 7_200 # 2 hours

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