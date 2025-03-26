# fail if any errors
set -ex

#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments.'

time {
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-jaxplan/step0_evaluate_fluents.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-jaxplan/step1_create_grounded_ablated_model.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-jaxplan/step2_create_baselise_run.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-jaxplan/step3_create_warm_start_policies_and_run.py
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-jaxplan/step4_compute_experiment_statistics.py
}