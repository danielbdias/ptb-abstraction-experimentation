#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run hyperparam tuning.'

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

time {
    PYTHONWARNINGS=ignore uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/_parameter_tuning.py
}