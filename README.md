# PtB Abstraction Experimentation

Experiments on how to create abstractions for RDDL to improve PtB performance.

## Development instructions

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
```

### Run script
```sh
PYTHONWARNINGS=ignore python ./experiment_file_.py
``````