#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# fail if any errors
set -e

# Function to get current timestamp in seconds
get_timestamp() {
    date +%s
}

# Function to calculate elapsed time
calculate_elapsed_time() {
    local start_time=$1
    local end_time=$2
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d" $minutes $seconds
}

# Function to print a title with decoration
print_title() {
    local title="$1"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo " $title "
    echo "--------------------------------------------------------------------------------"
    echo ""
}

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

DOMAINS=$2

export PYTHONWARNINGS=ignore # ignore warnings from the experiments

# Start timing the entire script
SCRIPT_START_TIME=$(get_timestamp)
print_title "Interval Analysis for $PLANNER_TYPE on $DOMAINS"

uv run python ./experiments/intervalanalysis-$PLANNER_TYPE/step1_evaluate_fluents.py --domains $DOMAINS

SCRIPT_END_TIME=$(get_timestamp)
print_title "Interval Analysis Complete"
print_title "Total execution time: $(calculate_elapsed_time $SCRIPT_START_TIME $SCRIPT_END_TIME)"
