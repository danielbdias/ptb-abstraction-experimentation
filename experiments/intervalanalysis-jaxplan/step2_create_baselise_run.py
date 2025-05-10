import os
import time
import jax

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxDeepReactivePolicy

from _config_run import experiments, jax_seeds, silent, run_drp, run_slp
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main, run_jax_planner
from _fileio import save_pickle_data

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, planner_type, experiment_params_builder):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - {planner_type}')

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################
    # domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    regular_environment = pyRDDLGym.make(domain=domain_instance_experiment.domain_name, instance=domain_instance_experiment.instance_name)
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain_instance_experiment.domain_name} (regular) - {planner_type}"

    for jax_seed in jax_seeds:
        experiment_params = experiment_params_builder(domain_instance_experiment)
        experiment_params.training_params.seed = jax.random.PRNGKey(jax_seed)
        
        experiment_summary = run_jax_planner(regular_experiment_name, rddl_model=grounded_model, planner_parameters=experiment_params, silent=silent)
        regular_env_experiment_stats.append(experiment_summary)

    save_pickle_data(regular_env_experiment_stats, f'{root_folder}/_results/baseline_{planner_type}_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pickle')

def drp_experiment_params_builder(domain_instance_experiment):
    experiment_params = domain_instance_experiment.drp_experiment_params
    experiment_params.optimizer_params.plan = JaxDeepReactivePolicy(domain_instance_experiment.drp_experiment_params.topology)
    return experiment_params

def slp_experiment_params_builder(domain_instance_experiment):
    experiment_params = domain_instance_experiment.slp_experiment_params
    experiment_params.optimizer_params.plan = JaxStraightLinePlan()
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
        if run_drp:
            args_list.append( (experiment, 'drp', drp_experiment_params_builder) )  
        if run_slp:
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