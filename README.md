# PtB Abstraction Experimentation

These are the experiments on how to create abstractions for RDDL to improve PtB performance.

Since we are using a slightly modified version (forked from the original ones) of the pyRDDLGym, pyRDDLGym-jax, and pyRDDLGym-gurobi, we need to install it locally using [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Development instructions

### Starting new env from scratch

```sh
# clone with submodules and go to folder
git clone --recurse-submodules git@github.com:danielbdias/ptb-abstraction-experimentation.git
cd ./ptb-abstraction-experimentation

# install dependencies
uv sync
```

### Run Experiments
```sh

sh ./_run_intervalanalysis_all_steps_slp.sh
```