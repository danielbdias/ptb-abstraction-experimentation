from _utils import load_data
from _graphs import stat_curves, plot_cost_curve_per_iteration
from _domains import domains

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_time_bars(title, planner, regular_stats, cold_start_stats, simplified_stats):
    regular_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, regular_stats)))
    cold_start_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, cold_start_stats)))
    simplified_mean_elapsed_time = np.mean(list(map(lambda item : item.elapsed_time, simplified_stats)))

    bar_names = (
        "PtB",
        "PtB-ColdStart",
    )
    time_sizes = {
        "Abstraction": np.array([ 0, simplified_mean_elapsed_time ]),
        "Original": np.array([ regular_mean_elapsed_time, cold_start_mean_elapsed_time ]),
    }
    width = 0.7

    plt.subplots(1, figsize=(7,8))
    bottom = np.zeros(2)

    for exp_type, total_time in time_sizes.items():
        plt.bar(bar_names, total_time, width, label=exp_type, bottom=bottom)
        bottom += total_time

    plt.title(title, fontsize=22, fontweight='bold')
    plt.ylabel("Time", fontsize=18)
    plt.legend(loc="best", fontsize=20)
    plt.tight_layout()

    #plt.rc('xtick', labelsize=30)
    plt.rcParams.update({'font.size':15})
    plt.rc('font', family='serif')

    plt.savefig(f'{root_folder}/_plots/{domain_name}_graph_time_{planner}.pdf', format='pdf')

print('--------------------------------------------------------------------------------')
print('Experiment Part 4 - Generating graphs for PtB with warm start')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)

for domain in domains:
    domain_name = domain.name

    statistics = {
        'Regular': load_data(f'{root_folder}/_results/{domain_name}_regular_statistics.pickle'),
        'Warm Start': load_data(f'{root_folder}/_results/{domain_name}_warmstart_statistics.pickle')
    }

    graph_path = f'{root_folder}/_plots/{domain_name}_slp.pdf'
    plot_cost_curve_per_iteration(f'Best Costs per Iteration ({domain_name})', statistics, lambda item : -item.best_return, graph_path)

    # TODO fix plot time to consider upper and lower bound
    # plot_time_bars(f'Mean execution time ({domain_name})', 'straightline', regular_straightline_stats, heuristic_straightline_stats, simplified_straightline_stats)

print('done!')