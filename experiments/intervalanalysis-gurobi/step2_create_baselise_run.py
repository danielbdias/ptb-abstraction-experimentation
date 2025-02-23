import os
import time

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from _config import experiments, silent
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main, run_gurobi_planner
from _fileio import save_pickle_data

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, planner_type):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - {planner_type}')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################
    _, domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_experiment_name = f"{domain_instance_experiment.domain_name} (regular) - {planner_type}"
    
    experiment_summary = run_gurobi_planner(regular_experiment_name, rddl_model=grounded_model, action_bounds=regular_environment._bounds, silent=silent)

    print(experiment_summary)

    save_pickle_data(experiment_summary, f'{root_folder}/_results/baseline_{planner_type}_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pickle')

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
        args_list.append( (experiment, 'slp') )
        
    # run experiment in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()