import csv
import jax
import optax
import time
import pickle

from dataclasses import dataclass

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym_jax.core.logic import FuzzyLogic
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxPlan

import numpy as np

@dataclass(frozen=True)
class PlanningModelParameters:
    logic:          FuzzyLogic

@dataclass(frozen=False)
class OptimizerParameters:
    plan:             JaxPlan
    optimizer:        optax.GradientTransformation
    learning_rate:    float
    batch_size_train: int
    batch_size_test:  int
    action_bounds:    dict

@dataclass(frozen=False)
class TrainingParameters:
    epochs:             int
    seed:               jax.random.PRNGKey
    train_seconds:      int
    policy_hyperparams: dict

@dataclass(frozen=False)
class PlannerParameters:
    model_params:               PlanningModelParameters
    optimizer_params:           OptimizerParameters   
    training_params:            TrainingParameters

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
        batch_size_train = planner_parameters.optimizer_params.batch_size_train,
        batch_size_test  = planner_parameters.optimizer_params.batch_size_test,
        plan             = planner_parameters.optimizer_params.plan,
        optimizer        = planner_parameters.optimizer_params.optimizer,
        optimizer_kwargs = {'learning_rate': planner_parameters.optimizer_params.learning_rate},
        action_bounds    = planner_parameters.optimizer_params.action_bounds)

    # run the planner as an optimization process
    planner_callbacks = planner.optimize(
        planner_parameters.training_params.seed, 
        epochs                 = planner_parameters.training_params.epochs, 
        policy_hyperparams     = planner_parameters.training_params.policy_hyperparams,
        train_seconds          = planner_parameters.training_params.train_seconds,
        return_callback        = True,
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