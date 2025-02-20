import optax
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

from typing import Any

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)
    
@dataclass(frozen=True)
class DomainInstanceExperiment:
    domain_name:                         str
    instance_name:                       str
    ground_fluents_to_freeze:            Set[str]
    bound_strategies:                    dict
    solver_timeout:                      int
    
    def __post_init__(self):
        if self.drp_experiment_params.topology is None:
            raise ValueError("drp_experiment_params must have a topology attribute set")
    
    def get_state_fluents(self, rddl_model):
        return list(rddl_model.state_fluents.keys())
    
    def get_experiment_paths(self, root_folder):
        domain_path = f"{root_folder}/domains/{self.domain_name}"
        domain_file_path = f'{domain_path}/domain.rddl'
        instance_file_path = f'{domain_path}/{self.instance_name}.rddl'
        
        return domain_path, domain_file_path, instance_file_path

@dataclass(frozen=True)
class ExperimentStatisticsSummary:
    final_solution:      dict
    accumulated_reward:  float
    elapsed_time:        float
    params:              Any


def run_gurobi_planner(name : str, rddl_model : RDDLPlanningModel, silent : bool = True):
    print(f'[{os.getpid()}] Run: {name} - Status: Starting')
    
    start_time = time.time()

    # initialize the planner and compiler
    planner = GurobiStraightLinePlan(rddl_model)
    compiler = GurobiRDDLCompiler(rddl=rddl_model, plan=planner, rollout_horizon=rddl_model.horizon)
    gurobi_environment = gurobipy.Env()

    # try to use the preconditions to produce narrow action bounds
    action_bounds = planner.action_bounds.copy()
    for name in compiler.rddl.action_fluents:
        if name not in action_bounds:
            action_bounds[name] = compiler.bounds.get(name, UNBOUNDED)

    planner.action_bounds = action_bounds

    # run the planner as an optimization process
    model, _, params = compiler.compile(env=gurobi_environment)
    model.optimize()
    
    print(model)
    
    solved = model.SolCount > 0 # check for existence of valid solution
    
    if not solved:
        raise ValueError("Gurobi failed to find a feasible solution")
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    if not silent:
        print(f'[{os.getpid()}] Run: {name} - Params: {params}')

    print(f'[{os.getpid()}] Run: {name} - Elapsed time: {elapsed_time:.2f} seconds')

    final_solution = {}
    for var in model.getVars():
        final_solution[var.varName] = var.X

    accumulated_reward = model.ObjVal

    return ExperimentStatisticsSummary(final_solution, accumulated_reward, elapsed_time, params)

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