from _utils import load_data
from _domains import domains
from _graphs import plot_cost_curve_per_iteration

import numpy as np

import csv
import os

def write_statistic_header(file_path):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Domain', 'State fluent', 'Mean (upper)', 'Mean (lower)', 'Mean (DiffMax)', 'DiffMax list'])

def diff_max(lower_bound_reward, upper_bound_reward):
    if upper_bound_reward < lower_bound_reward: # this means that our extremes are inverted
        new_upper_bound = lower_bound_reward
        lower_bound_reward = upper_bound_reward
        upper_bound_reward = new_upper_bound

    return (upper_bound_reward - lower_bound_reward)


def write_statistic_line(domain, state_fluent, lower_bound_statistics, upper_bound_statistics, file_path):
    best_returns_lower_bound_statistics = list(map(lambda item : item.statistics_history[-1].best_return, lower_bound_statistics))
    best_returns_upper_bound_statistics = list(map(lambda item : item.statistics_history[-1].best_return, upper_bound_statistics))

    number_of_experiments = len(best_returns_lower_bound_statistics)
    experiment_indexes = list(range(0, number_of_experiments))
    
    diff_max_rewards = list(map(lambda index : diff_max(best_returns_lower_bound_statistics[index], best_returns_upper_bound_statistics[index]), experiment_indexes))
    diff_max_rewards_as_string = list(map(lambda item : str(item), diff_max_rewards))

    best_returns_lower_bound_statistics_mean = np.mean(best_returns_lower_bound_statistics)
    best_returns_upper_bound_statistics_mean = np.mean(best_returns_upper_bound_statistics)
    diff_max_rewards_mean = np.mean(diff_max_rewards)
    diff_max_rewards_list = ', '.join(diff_max_rewards_as_string)

    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(['Domain', 'State fluent', 'Mean (upper)', 'Mean (lower)', 'DiffMax'])
        writer.writerow([domain, state_fluent, best_returns_lower_bound_statistics_mean, best_returns_upper_bound_statistics_mean, diff_max_rewards_mean, diff_max_rewards_list])

print('--------------------------------------------------------------------------------')
print('Experiment Part 2 - Generating graphs and tables for Sensibility Analysis')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)

statistic_file = f'{root_folder}/_plots/statistics_slp.csv'
write_statistic_header(statistic_file)

for domain in domains:
    domain_name = domain.name
    state_fluents = domain.state_fluents

    statistics = {}

    for state_fluent in state_fluents:
        statistics[f'Abstraction ({state_fluent} lower bound)'] = load_data(f'{root_folder}/_results/{domain_name}_abstraction_{state_fluent}_lower_bound_statistics.pickle')
        statistics[f'Abstraction ({state_fluent} upper bound)'] = load_data(f'{root_folder}/_results/{domain_name}_abstraction_{state_fluent}_upper_bound_statistics.pickle')

    graph_path = f'{root_folder}/_plots/{domain_name}_slp_curves.pdf'
    plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', statistics, lambda item : -item.best_return, graph_path)

    for state_fluent in state_fluents:
        lower_bound_statistics = statistics[f'Abstraction ({state_fluent} lower bound)']
        upper_bound_statistics = statistics[f'Abstraction ({state_fluent} upper bound)']
        write_statistic_line(domain.name, state_fluent, lower_bound_statistics, upper_bound_statistics, statistic_file)

print('done!')