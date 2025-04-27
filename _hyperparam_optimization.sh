#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run hyperparam optimization.'

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

time {
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/_parameter_tuning.py
}