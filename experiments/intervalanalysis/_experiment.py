import jax
import optax
import os
import time

from dataclasses import dataclass
from typing import List, Set
from multiprocessing import get_context, freeze_support

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym_jax.core.logic import ProductTNorm, FuzzyLogic, SigmoidComparison, SoftRounding, SoftControlFlow
from pyRDDLGym_jax.core.planner import JaxBackpropPlanner, JaxPlan, JaxPlannerStoppingRule

from _config_multiprocess import pool_context, num_workers, timeout

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
    guess:            dict

@dataclass(frozen=False)
class TrainingParameters:
    epochs:             int
    seed:               jax.random.PRNGKey
    train_seconds:      int
    policy_hyperparams: dict
    stopping_rule:      JaxPlannerStoppingRule

@dataclass(frozen=False)
class PlannerParameters:
    model_params:               PlanningModelParameters
    optimizer_params:           OptimizerParameters   
    training_params:            TrainingParameters
    topology:                   List[int] = None
    
    def is_drp(self):
        return self.topology is not None
    
    def is_slp(self):
        return not self.is_drp()

@dataclass(frozen=True)
class DomainInstanceExperiment:
    domain_name:                         str
    instance_name:                       str
    ground_fluents_to_freeze:            Set[str]
    bound_strategies:                    dict
    drp_experiment_params:               PlannerParameters
    slp_experiment_params:               PlannerParameters
    
    def __post_init__(self):
        if self.drp_experiment_params.topology is None:
            raise ValueError("drp_experiment_params must have a topology attribute set")
    
    def get_state_fluents(self, rddl_model):
        return list(rddl_model.state_fluents.keys())

def get_planner_parameters(model_weight : int, learning_rate : float, batch_size : int, epochs : int, policy_hyperparams: dict = None, topology : List[int] = None):
    return PlannerParameters(
        topology = topology,
        model_params = PlanningModelParameters(
            logic=FuzzyLogic(
                tnorm      = ProductTNorm(),
                comparison = SigmoidComparison(weight=model_weight),
                rounding   = SoftRounding(weight=model_weight),
                control    = SoftControlFlow(weight=model_weight),
            )
        ),
        optimizer_params = OptimizerParameters(
            plan             = None,
            optimizer        = optax.rmsprop,
            learning_rate    = learning_rate,
            batch_size_train = batch_size,
            batch_size_test  = batch_size,
            action_bounds    = None,
            guess            = None
        ),
        training_params = TrainingParameters(
            seed               = 42,
            epochs             = epochs,
            train_seconds      = 3_600,
            policy_hyperparams = policy_hyperparams,
            stopping_rule      = None
        )
    )

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

def run_jax_planner(name : str, rddl_model : RDDLPlanningModel, planner_parameters : PlannerParameters, silent : bool = True):
    print(f'[{os.getpid()}] Run: {name} - Status: starting')
    
    if not silent:
        print(f'[{os.getpid()}] Run: {name} - Seed: {planner_parameters.training_params.seed}')
    
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
    planner_callbacks = planner.optimize_generator(
        planner_parameters.training_params.seed, 
        epochs                 = planner_parameters.training_params.epochs, 
        policy_hyperparams     = planner_parameters.training_params.policy_hyperparams,
        train_seconds          = planner_parameters.training_params.train_seconds,
        guess                  = planner_parameters.optimizer_params.guess,
        print_summary          = not silent,
        print_progress         = not silent,
        stopping_rule          = planner_parameters.training_params.stopping_rule
    )

    final_policy_weights = None
    last_iteration_improved = None
    last_status = None
    statistics_history = []

    for callback in planner_callbacks:
        final_policy_weights = callback['best_params']
        last_iteration_improved = callback['last_iteration_improved']
        last_status = callback['status']

        statistics = ExperimentStatistics.from_callback(callback)
        statistics_history.append(statistics)

        if not silent:
            print(statistics)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'[{os.getpid()}] Ran: {name} - Status: {last_status} - Elapsed time: {elapsed_time:.2f} seconds')

    return ExperimentStatisticsSummary(final_policy_weights, statistics_history, elapsed_time, last_iteration_improved)

def prepare_parallel_experiment_on_main():
    freeze_support()
    
def run_experiment_in_parallel(perform_experiment_method, args_list):
    # create worker pool: note each iteration must wait for all workers
    # to finish before moving to the next
    with get_context(pool_context).Pool(processes=num_workers) as pool:
        multiple_results = [pool.apply_async(perform_experiment_method, args=args) for args in args_list]
        
        # wait for all workers to finish
        for res in multiple_results:
            res.get(timeout=timeout)