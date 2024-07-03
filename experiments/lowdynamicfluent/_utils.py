import os
from dataclasses import dataclass
from typing import Dict, List

import time

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent

from _domains import domains, DomainExperiment

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
    data_per_lifted_fluent: Dict[str, LiftedFluentData]

def run_single_simulation(domain_file_path : str, instance_file_path : str) -> SimulationData:
    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    
    # set up a random policy
    agent = RandomAgent(action_space=environment.action_space, num_actions=environment.max_allowed_actions)

    # perform a roll-out from the initial state
    state, _ = environment.reset()

    statistics = {}

    for ground_fluent_name in state.keys():
        fluent_data = GroundFluentData(fluent_name=ground_fluent_name, data_points=[])
        fluent_data.data_points.append(state[ground_fluent_name])
        statistics[ground_fluent_name] = fluent_data
    
    for _ in range(environment.horizon):
        environment.render()
        action = agent.sample_action(state)
        next_state, _, terminated, truncated, _ = environment.step(action)
        state = next_state

        # record data
        for ground_fluent_name in state.keys():
            fluent_data = statistics[ground_fluent_name]
            fluent_data.data_points.append(state[ground_fluent_name])

        if terminated or truncated:
            break

    return SimulationData(data_per_lifted_fluent=statistics)
