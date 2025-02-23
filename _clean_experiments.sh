#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to clean files.'

time {
    # clean intermediate files
    rm ./experiments/intervalanalysis-gurobi/_intermediate/*.model
    rm ./experiments/intervalanalysis-gurobi/_intermediate/*.rddl

    # clean plots
    rm ./experiments/intervalanalysis-gurobi/_plots/*.pdf

    # clean results
    rm ./experiments/intervalanalysis-gurobi/_results/*.csv
    rm ./experiments/intervalanalysis-gurobi/_results/*.pickle
}