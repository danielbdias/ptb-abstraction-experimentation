# PtB Abstraction Experimentation

These are the experiments on how to create abstractions for RDDL to improve PtB performance.

Since we are using a slightly modified version (forked from the original ones) of the pyRDDLGym and pyRDDLGym-jax, we need to install it locally.

## Development instructions

### Starting new env from scratch

```sh
# clone with submodules and go to folder
git clone --recurse-submodules git@github.com:danielbdias/ptb-abstraction-experimentation.git
cd ./ptb-abstraction-experimentation

# create venv
python -m venv ./_venv

# activate env
source _venv/bin/activate

# install requirements
pip install --upgrade pip
pip install -r requirements.txt

# install local pyRDDLGym libs
pip install ./pyRDDLGym
pip install ./pyRDDLGym-jax
```

### Run Experiments
```sh

sh ./_run_intervalanalysis_all_steps_slp.sh
```

### Install python with Tkinter on MacOSX
```sh
#based on https://stackoverflow.com/questions/59003269/getting-tkinter-to-work-with-python-3-x-on-macos-with-asdf
brew install python-tk@3.12

PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I/usr/local/opt/tcl-tk/include' --with-tcltk-libs='-L/usr/local/opt/tcl-tk/lib -ltcl8.6 -ltk8.6'" asdf install python 3.12.4
```