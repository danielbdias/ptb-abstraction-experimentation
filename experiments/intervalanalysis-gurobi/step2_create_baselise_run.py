import os
import time
import jax

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from _config import experiments, jax_seeds, silent, run_gurobi_planner
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main
from _fileio import save_pickle_data

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, planner_type, experiment_params_builder):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - {planner_type}')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################
    _, domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain_instance_experiment.domain_name} (regular) - {planner_type}"
    
    experiment_summary = run_gurobi_planner(regular_experiment_name, rddl_model=grounded_model, silent=silent)
    regular_env_experiment_stats.append(experiment_summary)

    save_pickle_data(regular_env_experiment_stats, f'{root_folder}/_results/baseline_{planner_type}_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pickle')

def slp_experiment_params_builder(domain_instance_experiment):
    experiment_params = domain_instance_experiment.slp_experiment_params
    return experiment_params

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
        args_list.append( (experiment, 'slp', slp_experiment_params_builder) )
        
    # run experiment in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()