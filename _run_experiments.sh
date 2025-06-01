# fail if any errors
set -ex

#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments.'

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

time {
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step0_evaluate_fluents.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step1_create_grounded_ablated_model.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step2_create_baselise_run.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step3_create_warm_start_policies_and_run.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step4_compute_experiment_statistics.py
}
