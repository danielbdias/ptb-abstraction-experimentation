import os
import time
import jax

from pyRDDLGym.core.grounder import RDDLGrounder

from _config_run import experiments, jax_seeds, silent, run_drp, run_slp, threshold_to_choose_fluents
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main, run_jax_planner
from _fileio import save_pickle_data, load_pickle_data, file_exists

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, strategy_name, threshold, planner_type, experiment_params_builder):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Threshold: {threshold} - Planner: {planner_type}')

    #########################################################################################################
    # Runs PtB with modified domain (that has ground fluents frozen with initial state values)
    #########################################################################################################

    # regular_domain_file_path, regular_instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'
    fluents_to_freeze_path = f"{root_folder}/_results/fluents_to_ablate_{file_common_suffix}.csv"
    if not file_exists(fluents_to_freeze_path):
        print(f'File for domain {domain_instance_experiment.domain_name} considering {strategy_name} strategy at threshold {threshold} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
        return

    regular_environment = domain_instance_experiment.get_pyrddlgym_environment(root_folder)
    grounder = RDDLGrounder(regular_environment.model.ast)
    regular_grounded_model = grounder.ground()
    
    ablated_model_file_path = f'{root_folder}/_intermediate/domain_{file_common_suffix}.model'
    ablated_grounded_model = load_pickle_data(ablated_model_file_path)

    warm_start_creation_experiment_stats = []
    warm_start_run_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params = experiment_params_builder(domain_instance_experiment)
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)

        env_params = experiment_params

        warm_start_creation_run_name = f"{domain_instance_experiment.domain_name} (creating warm-start) - {planner_type} - {strategy_name} - {threshold} - seed {jax_seed}"
        abstraction_env_experiment_summary = run_jax_planner(warm_start_creation_run_name, rddl_model=ablated_grounded_model, planner_parameters=env_params, silent=silent)
        warm_start_creation_experiment_stats.append(abstraction_env_experiment_summary)

        warm_start_env_params = experiment_params_builder(domain_instance_experiment, abstraction_env_experiment_summary.final_policy_weights)

        warm_start_run_name = f"{domain_instance_experiment.domain_name} (running with warm-start) - {planner_type} - {strategy_name} - {threshold} - seed {jax_seed}"
        warm_start_env_experiment_summary = run_jax_planner(warm_start_run_name, rddl_model=regular_grounded_model, planner_parameters=warm_start_env_params, silent=silent)
        warm_start_run_experiment_stats.append(warm_start_env_experiment_summary)

    save_pickle_data(warm_start_creation_experiment_stats, f'{root_folder}/_results/warmstart_creation_{planner_type}_run_data_{file_common_suffix}.pickle')
    save_pickle_data(warm_start_run_experiment_stats, f'{root_folder}/_results/warmstart_execution_{planner_type}_run_data_{file_common_suffix}.pickle')

def drp_experiment_params_builder(domain_instance_experiment, warm_start_policy=None):
    return domain_instance_experiment.drp_experiment_params_builder(warm_start_policy)
def slp_experiment_params_builder(domain_instance_experiment, warm_start_policy=None):
    return domain_instance_experiment.slp_experiment_params_builder(warm_start_policy)

if __name__ == '__main__':
    prepare_parallel_experiment_on_main()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create Warm Start policies and Run with Warm Start')
    print('--------------------------------------------------------------------------------')
    print()

    start_time = time.time()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    # create combination of parameters that we will use to run models
    args_list = []
    
    for domain_instance_experiment in experiments:
        for strategy_name in domain_instance_experiment.bound_strategies.keys():
            for threshold in threshold_to_choose_fluents:
                if run_drp:
                    args_list.append( (domain_instance_experiment, strategy_name, threshold, 'drp', drp_experiment_params_builder) )  
                if run_slp:
                    args_list.append( (domain_instance_experiment, strategy_name, threshold, 'slp', slp_experiment_params_builder) )
        
    # run experiment in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()