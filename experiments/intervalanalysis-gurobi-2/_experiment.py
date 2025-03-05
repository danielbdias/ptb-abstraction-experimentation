import os
import time

from dataclasses import dataclass
from typing import Set
from multiprocessing import get_context, freeze_support

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym_gurobi.core.compiler import GurobiRDDLCompiler
from pyRDDLGym_gurobi.core.planner import GurobiStraightLinePlan

from _config_multiprocess import pool_context, num_workers, timeout

import gurobipy
from gurobipy import GRB

import numpy as np

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)
    
@dataclass(frozen=True)
class DomainInstanceExperiment:
    domain_name:                         str
    instance_name:                       str
    ground_fluents_to_freeze:            Set[str]
    bound_strategies:                    dict
    solver_timeout:                      int
    
    def get_state_fluents(self, rddl_model):
        return list(rddl_model.state_fluents.keys())
    
    def get_experiment_paths(self, root_folder):
        domain_path = f"{root_folder}/domains/{self.domain_name}"
        domain_file_path = f'{domain_path}/domain.rddl'
        instance_file_path = f'{domain_path}/{self.instance_name}.rddl'
        
        return domain_path, domain_file_path, instance_file_path

@dataclass(frozen=True)
class ExperimentStatisticsSummary:
    final_solution:      list[dict]
    accumulated_reward:  float
    elapsed_time:        float

def compute_action_bounds(model, bounds):
    action_bounds = {}

    for action, prange in model.action_ranges.items():
        lower, upper = bounds[action]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        action_bounds[action] = (lower, upper)

    return action_bounds

def run_gurobi_planner(name : str, rddl_model : RDDLPlanningModel, action_bounds : dict, silent : bool = True):
    print(f'[{os.getpid()}] Run: {name} - Status: Starting')
    
    start_time = time.time()

    # initialize the planner and compiler
    updated_action_bounds = compute_action_bounds(rddl_model, action_bounds)
    
    planner = GurobiStraightLinePlan(updated_action_bounds)
    compiler = GurobiRDDLCompiler(rddl=rddl_model, plan=planner, rollout_horizon=rddl_model.horizon)
    gurobi_environment = gurobipy.Env()

    # run the planner as an optimization process
    model, _, params = compiler.compile(env=gurobi_environment)
    
    if warm_start_policy is not None:
        for step in range(rddl_model.horizon):
            for action, value in warm_start_policy[step].items():
                name = f'{action}__{step}'
                
                var = params[name][0]
                var.Start = value
    
    model.optimize()
    
    # check for existence of valid solution
    solved = model.SolCount > 0 
    
    if not solved:
        raise ValueError(f"Gurobi failed to find a feasible solution for experiment {name}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    if not silent:
        print(f'[{os.getpid()}] Run: {name} - Params: {params}')

    print(f'[{os.getpid()}] Run: {name} - Elapsed time: {elapsed_time:.2f} seconds')

    final_solution = []
    
    for step in range(rddl_model.horizon):
        action_values = planner.evaluate(compiler, params, step, {})
        final_solution.append(action_values)
        
    accumulated_reward = model.ObjVal

    return ExperimentStatisticsSummary(final_solution, accumulated_reward, elapsed_time)

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