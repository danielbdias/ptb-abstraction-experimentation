import csv
import os
import time

from _domains import domains

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis

from pyRDDLGym import RDDLEnv

import numpy as np

from collections import namedtuple
from typing import Tuple

BoudedTrajectory = namedtuple('BoundedTrajectory', ['fluent', 'reward_lower', 'reward_upper'])
BoundedAccumulatedReward = Tuple[float, float]

root_folder = os.path.dirname(__file__)

def record_time(file_path: str, time: float):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Time'])
        writer.writerow([time])

def record_reward_bounds_header(file_path: str):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            'Domain', 'Fluent', 
            'Accumulated Reward (LB)', 'Accumulated Reward (UB)', 
            'Score (Diff)', 'Score (Explained Interval)'
        ])

def compute_accumulated_reward(discount_factor : float, horizon: int, fluent_bounds: BoudedTrajectory) -> BoundedAccumulatedReward:
    accumulated_reward_lower_bound = 0.0
    accumulated_reward_upper_bound = 0.0

    for i in range(horizon):
        accumulated_reward_lower_bound += (discount_factor**i * fluent_bounds.reward_lower[i])
        accumulated_reward_upper_bound += (discount_factor**i * fluent_bounds.reward_upper[i])

    return (accumulated_reward_lower_bound, accumulated_reward_upper_bound)

def record_reward_values(file_path: str, domain_name: str, fluent_name : str, fluent_acc_reward: BoundedAccumulatedReward, regular_mdp_acc_reward: BoundedAccumulatedReward):
    accumulated_reward_lower_bound, accumulated_reward_upper_bound = fluent_acc_reward
    accumulated_reward_lower_bound_regular_mdp, accumulated_reward_upper_bound_regular_mdp = regular_mdp_acc_reward

    range_bounds = accumulated_reward_upper_bound - accumulated_reward_lower_bound
    range_bounds_regular_mdp = accumulated_reward_upper_bound_regular_mdp - accumulated_reward_lower_bound_regular_mdp

    score_diff = np.abs(range_bounds_regular_mdp - range_bounds)
    score_explained_interval = range_bounds / range_bounds_regular_mdp

    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            domain_name, fluent_name, 
            accumulated_reward_lower_bound, accumulated_reward_upper_bound, 
            score_diff, score_explained_interval
        ])

def build_ground_fluent_list(environment : RDDLEnv):
    ground_fluents = []
    
    for lifted_fluent in environment.model.state_fluents:
        params = environment.model.variable_params[lifted_fluent]
        grounded_objects = environment.model.ground_types(params)

        for object_name in grounded_objects:
            if len(object_name) == 0:
                ground_fluents.append(lifted_fluent)
            else:    
                ground_fluents.append(f"{lifted_fluent}___{object_name[0]}")

    return ground_fluents

def build_fluent_values_to_analyse(ground_fluent : str, state_bounds : dict, analysis : RDDLIntervalAnalysis):
    lifted_fluent = ground_fluent
    object_name = ''

    splitted_values = ground_fluent.split('___')

    if len(splitted_values) > 1:
        lifted_fluent, object_name = splitted_values[0], splitted_values[1]

    if object_name == '': # that means that there is no object to this fluent, just return the initial value
        return {lifted_fluent: state_bounds[lifted_fluent] }

    lower_values = analysis.rddl.state_fluents[lifted_fluent].copy()
    upper_values = analysis.rddl.state_fluents[lifted_fluent].copy()

    object_index = analysis.rddl.object_to_index[object_name]

    lower_values[object_index] = state_bounds[lifted_fluent][0][object_index]
    upper_values[object_index] = state_bounds[lifted_fluent][1][object_index]

    return {lifted_fluent: (np.asarray(lower_values), np.asarray(upper_values)) }

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

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Interval Analysis')
print('--------------------------------------------------------------------------------')
print()

# possible analysis - per grounded fluent, per lifted fluent
start_time = time.time()

#########################################################################################################
# This script will run interval propagation for each domain and instance, and record statistics
#########################################################################################################

for domain in domains:
    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'
    
    output_file_random_policy=f"{root_folder}/_results/intervals_table_random_policy_{domain.name}_{domain.instance}.csv"
    output_file_analysis_time=f"{root_folder}/_results/execution_time_random_policy_{domain.name}_{domain.instance}.csv"

    batch_size = domain.experiment_params.optimizer_params.batch_size_train

    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path, vectorized=True)

    discount_factor = environment.model.discount
    horizon = environment.model.horizon

    print(f'Domain: {domain.name} - Instance: {domain.instance}')

    # Random policy
    start_time_for_analysis = time.time()

    analysis = RDDLIntervalAnalysis(environment.model)
    action_bounds = compute_action_bounds(environment)
    state_bounds = compute_state_bounds(environment)

    # run first without fixing any fluent
    bounds = analysis.bound(action_bounds=action_bounds, per_epoch=True)
    regular_mdp_bounds = BoudedTrajectory('regular', bounds['reward'][0], bounds['reward'][1])
    regular_mdp_accumulated_reward = compute_accumulated_reward(discount_factor, horizon, regular_mdp_bounds)

    ground_fluents = build_ground_fluent_list(environment)
    ground_fluent_mdp_accumulated_reward = {}
    ground_fluent_initialization = {}

    for ground_fluent in ground_fluents:
        # test of fluent bounds 
        fluent_values = build_fluent_values_to_analyse(ground_fluent, state_bounds, analysis) # update initial state initialization
        ground_fluent_initialization[ground_fluent] = fluent_values
        
        # evaluate lower and upper bounds on accumulated reward of random policy
        bounds = analysis.bound(action_bounds=action_bounds, state_bounds=fluent_values, per_epoch=True)
        fixed_fluent_mdp_bounds = BoudedTrajectory(ground_fluent, bounds['reward'][0], bounds['reward'][1])
        fixed_fluent_mdp_accumulated_reward = compute_accumulated_reward(discount_factor, horizon, fixed_fluent_mdp_bounds)

        ground_fluent_mdp_accumulated_reward[ground_fluent] = fixed_fluent_mdp_accumulated_reward

    elapsed_time_for_analysis = time.time() - start_time_for_analysis
    record_time(output_file_analysis_time, elapsed_time_for_analysis)    

    print('Action bounds: ', action_bounds)
    print('Fluent bounds: ')
    for fluent_name, values in ground_fluent_initialization.items():
        print(f'  {fluent_name}: ', values)
    print()

    # generate score file
    record_reward_bounds_header(output_file_random_policy)
    record_reward_values(output_file_random_policy, domain.name, 'regular', regular_mdp_accumulated_reward, regular_mdp_accumulated_reward) # record regular MDP bounds

    for fluent_name in ground_fluent_mdp_accumulated_reward.keys():
        record_reward_values(output_file_random_policy, domain.name, fluent_name, ground_fluent_mdp_accumulated_reward[fluent_name], regular_mdp_accumulated_reward) # record fluent bounds

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()