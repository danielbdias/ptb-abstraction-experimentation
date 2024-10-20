#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run hyperparam optimization.'

time {
    # clean intermediate files
    rm ./experiments/intervalanalysis/_intermediate/*.model
    rm ./experiments/intervalanalysis/_intermediate/*.rddl

    # clean plots
    rm ./experiments/intervalanalysis/_plots/*.pdf

    # clean results
    rm ./experiments/intervalanalysis/_results/*.csv
    rm ./experiments/intervalanalysis/_results/*.pickle
}