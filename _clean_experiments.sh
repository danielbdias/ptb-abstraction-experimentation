#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to clean files.'

# receive the experiment name as an argument
PLANNER_TYPE=$1
PLANNER_TYPE=${PLANNER_TYPE:-jaxplan}

time {
    # clean intermediate files
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_intermediate/*.model
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_intermediate/*.rddl
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_intermediate/*.pickle

    # clean plots
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_plots/*.pdf

    # clean results
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_results/*.csv
    rm ./experiments/intervalanalysis-$PLANNER_TYPE/_results/*.pickle
}