# PtB Abstraction Experimentation

These are the experiments on how to create abstractions for RDDL to improve PtB performance.

Since we are using a slightly modified version (forked from the original ones) of the pyRDDLGym and pyRDDLGym-jax, we need to install it locally.

## Development instructions

### Clone repo

```sh
git clone --recurse-submodules git@github.com:danielbdias/ptb-abstraction-experimentation.git
```

### Create env 
```sh
python -m venv ./_venv
```

### Activate env 
```sh
source _venv/bin/activate
```

### Install requirements
```sh
pip install -r requirements.txt
```

### Install local pyRDDLGym libs
```sh
pip install ./pyRDDLGym
pip install ./pyRDDLGym-jax
pip install ./pyRDDLGym-symbolic
```

### Run script
```sh
PYTHONWARNINGS=ignore python ./experiment_file_.py
``````