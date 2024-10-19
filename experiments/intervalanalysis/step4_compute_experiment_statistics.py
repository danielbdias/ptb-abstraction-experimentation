from _utils import load_data
from _domains import domains

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence_time(plot_folder, domain_name, evaluation_time, warm_start_computation, warm_start_execution, baseline_execution):
    methods = (
        "Random Policy",
        "Warm Start Policy"
    )
    metrics = {
        "Fluent Eval Time": np.array([0, evaluation_time]),
        "Warm Start Computation": np.array([0, warm_start_computation]),
        "JaxPlan Execution": np.array([baseline_execution, warm_start_execution]),
    }
    width = 0.5

    y_axis_height = np.max([baseline_execution, evaluation_time + warm_start_computation + warm_start_execution]) * 1.4

    fig, ax = plt.subplots()
    bottom = np.zeros(2)

    for boolean, weight_count in metrics.items():
        p = ax.bar(methods, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(f"Convergence time ({domain_name})")
    ax.legend(loc="upper right", fontsize=14)

    plt.rcParams.update({'font.size':15})
    plt.rc('font', family='serif')

    plt.ylabel("Time (seconds)")
    plt.ylim(0, y_axis_height)

    plt.savefig(f'{plot_folder}/convergence_time_{domain_name}.pdf', format='pdf')


def read_fluent_evaluation_time_csv(file_path: str):
    lines = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            lines.append(row[0])

    return float(lines[1])

def plot_time_to_elect(plot_folder, domain_name, evaluation_time_low_fluent, warm_start_computation_low_fluent, evaluation_time_interval_analysis, warm_start_computation_interval_analysis):
    methods = (
        "Low Dynamic\nSimulation",
        "Interval Propagation"
    )
    metrics = {
        "Eval Time": np.array([evaluation_time_low_fluent, evaluation_time_interval_analysis]),
        "Warm Start Computation": np.array([warm_start_computation_low_fluent, warm_start_computation_interval_analysis]),
    }
    width = 0.5

    y_axis_height = np.max([evaluation_time_low_fluent + warm_start_computation_low_fluent, evaluation_time_interval_analysis + warm_start_computation_interval_analysis]) * 1.4

    fig, ax = plt.subplots()
    bottom = np.zeros(2)

    for boolean, weight_count in metrics.items():
        p = ax.bar(methods, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(f"Time to elect fluent ({domain_name})")
    ax.legend(loc="upper right", fontsize=14)

    plt.rcParams.update({'font.size':15})
    plt.rc('font', family='serif')

    plt.ylabel("Time (seconds)")
    plt.ylim(0, y_axis_height)

    plt.savefig(f'{plot_folder}/time_to_elect_fluent_{domain_name}.pdf', format='pdf')

def get_curves(experiment_summaries, attribute_getter):
    curves = []

    for experiment_summary in experiment_summaries:
        curve = np.array(list(map(attribute_getter, experiment_summary.statistics_history)))
        curves.append(curve)

    return curves

def stat_curves(experiment_summaries, attribute_getter):
    iteration_curves = get_curves(experiment_summaries, lambda item : item.iteration)
    curves = get_curves(experiment_summaries, attribute_getter)

    iteration_curve_max_len = -1
    larger_iteration_curve = None

    # find experiment with more iterations
    for i in range(len(experiment_summaries)):
        iteration_curve = iteration_curves[i]

        if len(iteration_curve) > iteration_curve_max_len:
            iteration_curve_max_len = len(iteration_curve)
            larger_iteration_curve = iteration_curve

    # repeat last value for each curve with less iterations
    resized_curves = []

    for i in range(len(experiment_summaries)):
        curve = curves[i]
        size_diff = iteration_curve_max_len - len(curve)
        if size_diff <= 0:
            resized_curves.append(curve)
        else:
            curve_last_value = curve[-1]
            resized_curve = np.append(curve, np.repeat(curve_last_value, size_diff))
            resized_curves.append(resized_curve)

    # convert "list of np.array" to "np.array of np.array"
    resized_curves = np.stack(resized_curves)

    curves_mean = np.mean(resized_curves, axis=0)
    curves_stddev = np.std(resized_curves, axis=0)

    return larger_iteration_curve, curves_mean, curves_stddev
    
def plot_cost_curve_per_iteration(plot_folder, domain_name, random_policy_stats, warm_start_stats):
    plt.subplots(1, figsize=(8,5))

    statistics = {
        'Random Policy': random_policy_stats,
        'Warm Start': warm_start_stats
    }

    for key in statistics.keys():
        stats = statistics[key]

        iterations, best_return_curves_mean, best_return_curves_stddev = stat_curves(stats, lambda item : item.best_return)
        
        plt.plot(iterations, best_return_curves_mean, '-', label=key)
        plt.fill_between(iterations, (best_return_curves_mean - best_return_curves_stddev), (best_return_curves_mean + best_return_curves_stddev), alpha=0.2)

    plt.title(f'Best Reward per Iteration ({domain_name})', fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend(loc="best", fontsize=14)
    plt.tight_layout()

    plt.rc('font', family='serif')

    plt.savefig(f'{plot_folder}/convergence_value_{domain_name}.pdf', format='pdf')

print('--------------------------------------------------------------------------------')
print('Experiment Part 4 - Generating graphs for PtB with warm start')
print('--------------------------------------------------------------------------------')
print()

root_folder = os.path.dirname(__file__)
plot_folder = f'{root_folder}/_plots'

for domain in domains:
    domain_name = domain.name

    ############################################################
    # Convergence value
    ############################################################

    random_policy_stats = load_data(f'{root_folder}/_results/baseline_run_data_{domain_name}_{domain.instance}.pickle')
    warm_start_stats = load_data(f'{root_folder}/_results/warmstart_execution_run_data_{domain_name}_{domain.instance}.pickle')   

    plot_cost_curve_per_iteration(plot_folder, domain_name, random_policy_stats, warm_start_stats)

    ############################################################
    # Convergence time
    ############################################################

    warm_start_creation_experiment_stats = load_data(f'{root_folder}/_results/warmstart_creation_run_data_{domain_name}_{domain.instance}.pickle')
    warm_start_execution_experiment_stats = load_data(f'{root_folder}/_results/warmstart_execution_run_data_{domain_name}_{domain.instance}.pickle')
    baseline_execution_experiment_stats = load_data(f'{root_folder}/_results/baseline_run_data_{domain_name}_{domain.instance}.pickle')

    evaluation_time = read_fluent_evaluation_time_csv(f'{root_folder}/_results/execution_time_random_policy_{domain_name}_{domain.instance}.csv')
    warm_start_computation = np.mean(list(map(lambda item : item.elapsed_time, warm_start_creation_experiment_stats)))
    warm_start_execution = np.mean(list(map(lambda item : item.elapsed_time, warm_start_execution_experiment_stats)))
    baseline_execution = np.mean(list(map(lambda item : item.elapsed_time, baseline_execution_experiment_stats)))

    plot_convergence_time(plot_folder, domain_name, evaluation_time, warm_start_computation, warm_start_execution, baseline_execution)

    ############################################################
    # Time to elect fluent
    ############################################################

    warm_start_creation_experiment_stats_low_fluent = load_data(f'{root_folder}/_results/warmstart_creation_run_data_{domain_name}_{domain.instance}.pickle')

    # metrics for low fluent
    warm_start_computation_low_fluent = np.mean(list(map(lambda item : item.elapsed_time, warm_start_creation_experiment_stats_low_fluent)))
    evaluation_time_low_fluent = read_fluent_evaluation_time_csv(f'{root_folder}/_results/execution_time_random_policy_{domain_name}_{domain.instance}.csv')

    # metrics for interval analysis
    evaluation_time_interval_analysis = 0
    warm_start_computation_interval_analysis = 0

    plot_time_to_elect(plot_folder, domain_name, evaluation_time_low_fluent, warm_start_computation_low_fluent, evaluation_time_interval_analysis, warm_start_computation_interval_analysis)

print('done!')