#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# fail if any errors
set -ex

#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments.'

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

export PYTHONWARNINGS=ignore # ignore warnings from the experiments

time {
    uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step0_evaluate_fluents.py
    uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step1_create_grounded_ablated_model.py
    uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step2_create_baselise_run.py
    uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step3_create_warm_start_policies_and_run.py
    uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step4_compute_experiment_statistics.py
}
