import os
import time
import traceback
import pyRDDLGym

from _config import experiments, silent
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main, run_gurobi_planner
from _fileio import save_pickle_data

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name}')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################
    _, domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path, vectorized=True)

    regular_experiment_name = f"{domain_instance_experiment.domain_name} (regular)"
    
    try:
        experiment_summary = run_gurobi_planner(regular_experiment_name, rddl_model=regular_environment.model, action_bounds=regular_environment._bounds, silent=silent)
        save_pickle_data(experiment_summary, f'{root_folder}/_results/baseline_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pickle')
    except Exception as e:
        print(f"Error running experiment {regular_experiment_name}: {e}")
        traceback.print_exc()
        return

if __name__ == '__main__':
    prepare_parallel_experiment_on_main()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create baseline Run with Random policy')
    print('--------------------------------------------------------------------------------')
    print()

    start_time = time.time()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    # create combination of parameters that we will use to run baseline models
    args_list = []
    
    for experiment in experiments:
        args_list.append( (experiment, ) )
        
    # run experiment in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()