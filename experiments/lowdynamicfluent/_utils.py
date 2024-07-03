from dataclasses import dataclass
from typing import Dict, List

import pyRDDLGym
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.policy import RandomAgent

import numpy

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
    data_per_lifted_fluent: Dict[str, LiftedFluentData]

def find_lifted_fluent(ground_fluent_name: str, lifted_fluents: List[str]) -> str:
    for lifted_fluent in lifted_fluents:
        if ground_fluent_name.startswith(lifted_fluent):
            return lifted_fluent
    
    return ""

def run_simulations(domain_file_path : str, instance_file_path : str, lifted_fluents: List[str], number_of_simulations: int) -> List[SimulationData]:
    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    simulations = []

    for i in range(number_of_simulations):
        simulations.append(run_single_simulation(environment, lifted_fluents, i)) 

    return simulations

def convert_to_number(value):
    if type(value) != numpy.bool_:
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

    return SimulationData(data_per_lifted_fluent=statistics, simulation_number=simulation_number)
