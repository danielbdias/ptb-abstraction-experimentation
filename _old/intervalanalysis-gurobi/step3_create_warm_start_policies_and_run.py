import os
import time

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from _config import experiments, silent, threshold_to_choose_fluents
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main, run_gurobi_planner
from _fileio import save_pickle_data, load_pickle_data, file_exists

root_folder = os.path.dirname(__file__)


def perform_experiment(domain_instance_experiment, strategy_name, threshold):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Threshold: {threshold}')

    #########################################################################################################
    # Runs PtB with modified domain (that has ground fluents frozen with initial state values)
    #########################################################################################################
    try:
        _, regular_domain_file_path, regular_instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

        file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'
        fluents_to_freeze_path = f"{root_folder}/_results/fluents_to_ablate_{file_common_suffix}.csv"
        if not file_exists(fluents_to_freeze_path):
            print(f'File for domain {domain_instance_experiment.domain_name} considering {strategy_name} strategy at threshold {threshold} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
            return

        regular_environment = pyRDDLGym.make(domain=regular_domain_file_path, instance=regular_instance_file_path)
        grounder = RDDLGrounder(regular_environment.model.ast)
        regular_grounded_model = grounder.ground()
        
        ablated_model_file_path = f'{root_folder}/_intermediate/domain_{file_common_suffix}.model'
        ablated_grounded_model = load_pickle_data(ablated_model_file_path)

        warm_start_creation_run_name = f"{domain_instance_experiment.domain_name} (creating warm-start) - {strategy_name} - {threshold}"
        abstraction_env_experiment_summary = run_gurobi_planner(
            warm_start_creation_run_name, 
            rddl_model=ablated_grounded_model, 
            action_bounds=regular_environment._bounds, 
            silent=silent
        )

        warm_start_run_name = f"{domain_instance_experiment.domain_name} (running with warm-start) - {strategy_name} - {threshold}"
        warm_start_env_experiment_summary = run_gurobi_planner(
            warm_start_run_name, 
            rddl_model=regular_grounded_model, 
            action_bounds=regular_environment._bounds, 
            warm_start_policy=abstraction_env_experiment_summary.final_solution, 
            silent=silent
        )

        save_pickle_data(abstraction_env_experiment_summary, f'{root_folder}/_results/warmstart_creation_run_data_{file_common_suffix}.pickle')
        save_pickle_data(warm_start_env_experiment_summary, f'{root_folder}/_results/warmstart_execution_run_data_{file_common_suffix}.pickle')
    except Exception as e:
        print(f"Error running experiment for domain {domain_instance_experiment.domain_name} and instance {domain_instance_experiment.instance_name}: {e}")
        return


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
                args_list.append( (domain_instance_experiment, strategy_name, threshold, ) )  
        
    # run experiment in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()