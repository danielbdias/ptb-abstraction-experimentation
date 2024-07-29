import csv
import jax
import optax
import time
import pickle

import pyRDDLGym.core
from scipy.stats import entropy

from dataclasses import dataclass
from typing import Dict, List, Set

import pyRDDLGym
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.policy import RandomAgent
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxPlan, JaxOfflineController

import numpy as np

@dataclass(frozen=True)
class GroundFluentData:
    fluent_name:        str
    data_points:        List

@dataclass(frozen=True)
class LiftedFluentData:
    fluent_name:        str
    ground_fluent_data: Dict[str, GroundFluentData]

@dataclass(frozen=True)
class SimulationData:
    simulation_number:      int
    lifted_fluent_data:     Dict[str, LiftedFluentData]

@dataclass(frozen=True)
class FluentValueStatististic:
    fluent_name:        str
    mean:               float
    standard_deviation: float
    variance:           float
    entropy:            float

@dataclass(frozen=True)
class PlannerParameters:
    batch_size_train:           int
    plan:                       JaxPlan
    optimizer:                  optax.GradientTransformation
    learning_rate:              float
    epochs:                     int
    seed:                       jax.random.PRNGKey
    action_bounds:              dict
    epsilon_error:              float
    epsilon_iteration_stop:     int
    policy_hyperparams:         dict
    ground_fluents_to_freeze:   Set[str]

def find_lifted_fluent(ground_fluent_name: str, lifted_fluents: List[str]) -> str:
    for lifted_fluent in lifted_fluents:
        if ground_fluent_name.startswith(lifted_fluent):
            return lifted_fluent
    
    return ""

def convert_to_number(value):
    if type(value) != np.bool_:
        return value
    
    if value:
        return 1.0
    
    return -1.0

def add_state_data_to_statistics(state: Dict, statistics: Dict, ground_to_lifted_names: List[str]):
    for ground_fluent_name in state.keys():
        lifted_fluent_name = ground_to_lifted_names[ground_fluent_name]

        if lifted_fluent_name not in statistics.keys():
            statistics[lifted_fluent_name] = LiftedFluentData(lifted_fluent_name, {})

        lifted_fluent_data = statistics[lifted_fluent_name]

        if ground_fluent_name not in lifted_fluent_data.ground_fluent_data.keys():
            lifted_fluent_data.ground_fluent_data[ground_fluent_name] = GroundFluentData(ground_fluent_name, [])

        ground_fluent_data = lifted_fluent_data.ground_fluent_data[ground_fluent_name]

        value = convert_to_number(state[ground_fluent_name])
        ground_fluent_data.data_points.append(value)

def run_single_simulation(environment : RDDLEnv, lifted_fluents: List[str], simulation_number: int) -> SimulationData:
    # set up a random policy
    agent = RandomAgent(action_space=environment.action_space, num_actions=environment.max_allowed_actions)

    # perform a roll-out from the initial state
    state, _ = environment.reset()

    statistics = {}
    ground_to_lifted_names = {}

    # build name cache
    for ground_fluent_name in state.keys():
        lifted_fluent_name = find_lifted_fluent(ground_fluent_name, lifted_fluents)
        ground_to_lifted_names[ground_fluent_name] = lifted_fluent_name

    # record data for initial state
    add_state_data_to_statistics(state, statistics, ground_to_lifted_names)
    
    for _ in range(environment.horizon):
        action = agent.sample_action(state)
        next_state, _, terminated, truncated, _ = environment.step(action)
        state = next_state

        # record data
        add_state_data_to_statistics(state, statistics, ground_to_lifted_names)

        if terminated or truncated:
            break

    return SimulationData(lifted_fluent_data=statistics, simulation_number=simulation_number)

def run_simulations(environment: pyRDDLGym.RDDLEnv, lifted_fluents: List[str], number_of_simulations: int) -> List[SimulationData]:
    simulations = []

    for i in range(number_of_simulations):
        simulations.append(run_single_simulation(environment, lifted_fluents, i)) 

    return simulations

def aggregate_simulation_data(data : List[SimulationData]):
    lifted_fluent_data = {}
    ground_fluent_data = {}

    # initialize lists
    first_lifted_fluent_data = data[0].lifted_fluent_data 
    for lifted_fluent_name in first_lifted_fluent_data.keys():
        lifted_fluent_data[lifted_fluent_name] = []
        for ground_fluent_name in first_lifted_fluent_data[lifted_fluent_name].ground_fluent_data.keys():
            ground_fluent_data[ground_fluent_name] = []

    # aggregate series
    for simulation_data in data:
        for lifted_fluent_name in simulation_data.lifted_fluent_data.keys():
            for ground_fluent_name in simulation_data.lifted_fluent_data[lifted_fluent_name].ground_fluent_data.keys():
                data_points = simulation_data.lifted_fluent_data[lifted_fluent_name].ground_fluent_data[ground_fluent_name].data_points
                
                lifted_fluent_data[lifted_fluent_name] += data_points
                ground_fluent_data[ground_fluent_name] += data_points

    return lifted_fluent_data, ground_fluent_data

def compute_fluent_stats(fluent_name, data, entropy_bin):
    return FluentValueStatististic(
        fluent_name=fluent_name,
        mean=np.mean(data),
        standard_deviation=np.std(data),
        variance=np.var(data),
        entropy=compute_entropy(data, entropy_bin)
    )

def compute_entropy(data, entropy_bin):
    _, bins = np.histogram(data, bins=entropy_bin)
    return entropy(bins)

def compute_statistics(data : List[SimulationData], entropy_bin) -> List[FluentValueStatististic]:
    lifted_fluent_data, ground_fluent_data = aggregate_simulation_data(data)

    result = []

    for key in lifted_fluent_data.keys():
        result.append(compute_fluent_stats(key, lifted_fluent_data[key], entropy_bin))

    for key in ground_fluent_data.keys():
        result.append(compute_fluent_stats(key, ground_fluent_data[key], entropy_bin))

    return result

def record_csv(file_path: str, domain_name: str, data: List[FluentValueStatististic]):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Domain', 'Fluent', 'Mean', 'StdDev', 'Variance', 'Entropy'])

        for item in data:
            writer.writerow([domain_name, item.fluent_name, item.mean, item.standard_deviation, item.variance, item.entropy])
    
def run_jaxplanner_with_record(name, environment, planner_parameters, silent=True):
    if not silent:
        print('--------------------------------------------------------------------------------')
        print('Domain: ', name)
        print('Seed: ', planner_parameters.seed)
        print('--------------------------------------------------------------------------------')
        print()
    
    start_time = time.time()

    # initialize the planner
    planner = JaxBackpropPlanner(
        environment.model,
        batch_size_train=planner_parameters.batch_size_train,
        plan=planner_parameters.plan,
        optimizer=planner_parameters.optimizer,
        optimizer_kwargs={'learning_rate': planner_parameters.learning_rate},
        action_bounds=planner_parameters.action_bounds)

    # run the planner as an optimization process
    planner_callbacks = planner.optimize(
        planner_parameters.seed, 
        epochs=planner_parameters.epochs, 
        epsilon_error=planner_parameters.epsilon_error,
        epsilon_iteration_stop=planner_parameters.epsilon_iteration_stop,
        policy_hyperparams=planner_parameters.policy_hyperparams,
        return_callback=True,
        record_training_batches=True,
    )

    values_per_state_variable = {}

    for callback in planner_callbacks:
        training_batches = callback['training_batches']

        for key in training_batches.keys():
            if key not in values_per_state_variable.keys():
                values_per_state_variable[key] = []

            values_per_state_variable[key] += training_batches[key]

        if not silent:
            print(callback)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return values_per_state_variable, elapsed_time

def aggregate_and_fix_jax_simulations(jax_simulations, lifted_fluents):
    ground_fluent_data = {}

    for simulation in jax_simulations:
        for ground_fluent_name in simulation.keys():
            if ground_fluent_name not in ground_fluent_data.keys():
                ground_fluent_data[ground_fluent_name] = []
            ground_fluent_data[ground_fluent_name] += simulation[ground_fluent_name]

    # add lifted values
    lifted_fluent_data = {}
    for ground_fluent_name in ground_fluent_data.keys():
        lifted_fluent_name = find_lifted_fluent(ground_fluent_name, lifted_fluents)
        if lifted_fluent_name not in lifted_fluent_data.keys():
            lifted_fluent_data[lifted_fluent_name] = []
        lifted_fluent_data[lifted_fluent_name] += ground_fluent_data[ground_fluent_name]
        
    return lifted_fluent_data, ground_fluent_data

def compute_jax_simulation_statistics(jax_simulations, lifted_fluents, entropy_bin):
    lifted_fluent_data, ground_fluent_data = aggregate_and_fix_jax_simulations(jax_simulations, lifted_fluents)

    result = []

    for key in lifted_fluent_data.keys():
        result.append(compute_fluent_stats(key, lifted_fluent_data[key], entropy_bin))

    for key in ground_fluent_data.keys():
        result.append(compute_fluent_stats(key, ground_fluent_data[key], entropy_bin))

    return result

def run_jaxplanner(name, environment, planner_parameters, silent=True):
    if not silent:
        print('--------------------------------------------------------------------------------')
        print('Domain: ', name)
        print('Seed: ', planner_parameters.seed)
        print('--------------------------------------------------------------------------------')
        print()
    
    start_time = time.time()

    # initialize the planner
    planner = JaxBackpropPlanner(
        environment.model,
        batch_size_train=planner_parameters.batch_size_train,
        plan=planner_parameters.plan,
        optimizer=planner_parameters.optimizer,
        optimizer_kwargs={'learning_rate': planner_parameters.learning_rate},
        action_bounds=planner_parameters.action_bounds,
        ground_fluents_to_freeze=planner_parameters.ground_fluents_to_freeze
    )

    # run the planner as an optimization process
    planner_callbacks = planner.optimize(
        planner_parameters.seed, 
        epochs=planner_parameters.epochs, 
        epsilon_error=planner_parameters.epsilon_error,
        epsilon_iteration_stop=planner_parameters.epsilon_iteration_stop,
        policy_hyperparams=planner_parameters.policy_hyperparams,
        return_callback=True,
    )

    # final_policy_weights = {}

    for callback in planner_callbacks:
        # final_policy_weights = callback['best_params']

        if not silent:
            print(callback)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return planner, elapsed_time

def run_single_simulation_with_policy(environment : RDDLEnv, lifted_fluents: List[str], policy: JaxBackpropPlanner, jax_seed, simulation_number: int) -> SimulationData:
    agent = JaxOfflineController(policy, key=jax.random.PRNGKey(jax_seed))

    # perform a roll-out from the initial state
    state, _ = environment.reset()

    statistics = {}
    ground_to_lifted_names = {}

    # build name cache
    for ground_fluent_name in state.keys():
        lifted_fluent_name = find_lifted_fluent(ground_fluent_name, lifted_fluents)
        ground_to_lifted_names[ground_fluent_name] = lifted_fluent_name

    # record data for initial state
    add_state_data_to_statistics(state, statistics, ground_to_lifted_names)
    
    for _ in range(environment.horizon):
        action = agent.sample_action(state)
        next_state, _, terminated, truncated, _ = environment.step(action)
        state = next_state

        # record data
        add_state_data_to_statistics(state, statistics, ground_to_lifted_names)

        if terminated or truncated:
            break

    return SimulationData(lifted_fluent_data=statistics, simulation_number=simulation_number)

def run_simulations_with_policy(environment: pyRDDLGym.RDDLEnv, lifted_fluents: List[str], policy: JaxBackpropPlanner, jax_seed, number_of_simulations: int) -> List[SimulationData]:
    simulations = []

    for i in range(number_of_simulations):
        simulations.append(run_single_simulation_with_policy(environment, lifted_fluents, policy, jax_seed, i)) 

    return simulations

########################################
########################################
########################################

@dataclass(frozen=True)
class ExperimentStatistics:
    iteration:                   int
    train_return:                float
    test_return:                 float
    best_return:                 float

    @staticmethod
    def from_callback(planner_callback):
        iteration = planner_callback['iteration']

        # possible keys: 'iteration', 'train_return', 'test_return', 'best_return', 'params', 'best_params', 'last_iteration_improved', 'grad', 'updates', 'action', 'error', 'invariant', 'precondition', 'pvar', 'reward', 'termination'
        return ExperimentStatistics(
            iteration=iteration,
            train_return=planner_callback['train_return'],
            test_return=planner_callback['test_return'],
            best_return=planner_callback['best_return'],
        )
    
    def __str__(self) -> str:
        return 'step={} train_return={:.6f} test_return={:.6f} best_return={:.6f}'.format(
          str(self.iteration).rjust(4), self.train_return, self.test_return, self.best_return)

@dataclass(frozen=True)
class ExperimentStatisticsSummary:
    final_policy_weights:        dict
    statistics_history:          list
    elapsed_time:                float
    last_iteration_improved:     int

def run_experiment(name : str, rddl_model : RDDLPlanningModel, planner_parameters : PlannerParameters, silent : bool = True):
    if not silent:
        print('--------------------------------------------------------------------------------')
        print('Experiment: ', name)
        print('Seed: ', planner_parameters.seed)
        print('--------------------------------------------------------------------------------')
        print()
    
    start_time = time.time()

    # initialize the planner
    planner = JaxBackpropPlanner(
        rddl_model,
        batch_size_train=planner_parameters.batch_size_train,
        plan=planner_parameters.plan,
        optimizer=planner_parameters.optimizer,
        optimizer_kwargs={'learning_rate': planner_parameters.learning_rate},
        action_bounds=planner_parameters.action_bounds)

    # run the planner as an optimization process
    planner_callbacks = planner.optimize(
        planner_parameters.seed, 
        epochs=planner_parameters.epochs, 
        epsilon_error=planner_parameters.epsilon_error,
        epsilon_iteration_stop=planner_parameters.epsilon_iteration_stop,
        policy_hyperparams=planner_parameters.policy_hyperparams,
        return_callback=True,
    )

    final_policy_weights = None
    last_iteration_improved = None
    statistics_history = []

    for callback in planner_callbacks:
        final_policy_weights = callback['best_params']
        last_iteration_improved = callback['last_iteration_improved']

        statistics = ExperimentStatistics.from_callback(callback)
        statistics_history.append(statistics)

        if not silent:
            print(statistics)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return ExperimentStatisticsSummary(final_policy_weights, statistics_history, elapsed_time, last_iteration_improved)

def save_data(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
    
def save_time(experiment_name, time, file_path):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([experiment_name, time])

def load_time_csv(file_path):
    result = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for record in reader:
            experiment_name, time = record
            result[experiment_name] = float(time)

    return result