import os
import time
import csv
import numpy as np

import pyRDDLGym
from pyRDDLGym import RDDLEnv

from pyRDDLGym.core.intervals import RDDLIntervalAnalysis, RDDLIntervalAnalysisMean, RDDLIntervalAnalysisPercentile

from _config import experiments, threshold_to_choose_fluents
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main

from _fileio import file_exists, get_ground_fluents_to_ablate_from_csv, save_pickle_data

def record_time(file_path: str, time: float):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Time'])
        writer.writerow([time])

def get_ground_fluents_to_ablate(domain, file_path: str):
    if domain.ground_fluents_to_freeze is not None and len(domain.ground_fluents_to_freeze) > 0:
        return domain.ground_fluents_to_freeze

    return get_ground_fluents_to_ablate_from_csv(file_path)

def get_interval_analysis(environment: RDDLEnv, strategy_type: str, strategy_params: dict):
    if strategy_type == 'mean':
        return RDDLIntervalAnalysisMean(environment)
    elif strategy_type == 'percentile':
        return RDDLIntervalAnalysisPercentile(environment, strategy_params['percentiles'])
    else:
        return RDDLIntervalAnalysis(environment)

def compute_action_bounds(environment):
    action_bounds = {}

    for action, prange in environment.model.action_ranges.items():
        lower, upper = environment._bounds[action]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        action_bounds[action] = (lower, upper)

    return action_bounds

def compute_state_bounds(environment : RDDLEnv):
    state_bounds = {}

    for state_lifted_fluent, prange in environment.model.state_ranges.items():
        lower, upper = environment._bounds[state_lifted_fluent]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        state_bounds[state_lifted_fluent] = (lower, upper)

    return state_bounds

def build_fluent_values_to_analyse(ground_fluents : str, state_bounds : dict, analysis : RDDLIntervalAnalysis):
    fluent_values = state_bounds.copy()
    
    for ground_fluent in ground_fluents:
        lifted_fluent = ground_fluent
        object_name = ''

        splitted_values = ground_fluent.split('___')

        if len(splitted_values) > 1:
            lifted_fluent, object_name = splitted_values[0], splitted_values[1]

        if object_name == '': # that means that there is no object to this fluent, just return the initial value
            state_bounds[lifted_fluent] = state_bounds[ground_fluent]

        lower_values = analysis.rddl.state_fluents[lifted_fluent].copy()
        upper_values = analysis.rddl.state_fluents[lifted_fluent].copy()

        object_index = analysis.rddl.object_to_index[object_name]

        lower_values[object_index] = state_bounds[lifted_fluent][0][object_index]
        upper_values[object_index] = state_bounds[lifted_fluent][1][object_index]

        fluent_values[lifted_fluent] = (np.asarray(lower_values), np.asarray(upper_values))

    return fluent_values

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, strategy_name, strategy, threshold):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Threshold: {threshold}')
    
    _, domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'
    
    fluents_to_freeze_path = f"{root_folder}/_results/fluents_to_ablate_{file_common_suffix}.csv"
    output_file_final_analysis_time=f"{root_folder}/_results/time_final_analysis_{file_common_suffix}.csv"
    
    if not file_exists(fluents_to_freeze_path):
        print(f'File for domain {domain_instance_experiment.domain_name} considering {strategy_name} strategy at threshold {threshold} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
        return

    # Random policy
    start_time_for_analysis = time.time()

    fluents_to_ablate = get_ground_fluents_to_ablate(domain_instance_experiment, fluents_to_freeze_path)
    
    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path, vectorized=True)
    analysis = get_interval_analysis(environment.model, strategy_name, strategy)
    
    action_bounds = compute_action_bounds(environment)
    state_bounds = compute_state_bounds(environment)
    
    fluent_values = build_fluent_values_to_analyse(fluents_to_ablate, state_bounds, analysis)
    
    bounds = analysis.bound(action_bounds=action_bounds, state_bounds=fluent_values, per_epoch=True)
    
    action_bounds_to_save = {}
    for action_name in action_bounds.keys():
        action_bounds_to_save[action_name] = bounds[action_name]

    elapsed_time_for_analysis = time.time() - start_time_for_analysis
    
    record_time(output_file_final_analysis_time, elapsed_time_for_analysis)
    save_pickle_data(action_bounds_to_save, f'{root_folder}/_intermediate/action_bounds_{file_common_suffix}.pickle')

if __name__ == '__main__':
    prepare_parallel_experiment_on_main()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create Action Bounds for Ablated Fluents')
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    start_time = time.time()

    # create combination of parameters that we will use to create ground models
    args_list = []
    
    for domain_instance_experiment in experiments:
        for strategy_name, strategy in domain_instance_experiment.bound_strategies.items():
            for threshold in threshold_to_choose_fluents:
                args_list.append( (domain_instance_experiment, strategy_name, strategy, threshold) )

    # Run experiments in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()