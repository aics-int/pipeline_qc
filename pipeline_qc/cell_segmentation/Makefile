# MAKE defaults
# See https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

##############################################
PYTHON_VERSION=3.7

VENV_LOCAL:=$(shell pwd)/venv/local
VENV_V100:=$(shell pwd)/venv/v100
VENV_TITAN:=$(shell pwd)/venv/titan
CONDA=conda


.PHONY: install-local install-v100 install-titan clean activate

## local
venv-local: $(VENV_LOCAL)/bin/python
$(VENV_LOCAL)/bin/python:
> test -d $(VENV_LOCAL) || $(CONDA) create -y -p $(VENV_LOCAL) python=$(PYTHON_VERSION)
> $(VENV_LOCAL)/bin/python -m pip install -U pip

install-local: venv-local
> $(VENV_LOCAL)/bin/python -m pip install --src $(VENV_LOCAL)/src -U -r requirements.txt -e ../..[all]
> cd $(VENV_LOCAL)/src/cell-detector/cell_detector/bin/
> rm -rf build
> $(VENV_LOCAL)/bin/python setup.py build develop
> cd $(VENV_LOCAL)/src/cell-detector/cell_detector/bin/apex/
> rm -rf build
> $(VENV_LOCAL)/bin/python setup.py install --cuda_ext --cpp_ext

## v100
venv-v100: $(VENV_V100)/bin/python
$(VENV_V100)/bin/python:
> test -d $(VENV_V100) || $(CONDA) create -y -p $(VENV_V100) python=$(PYTHON_VERSION)
> $(VENV_V100)/bin/python -m pip install -U pip

install-v100: venv-v100
> $(VENV_V100)/bin/python -m pip install --src $(VENV_V100)/src -U -r requirements.cu101.v100.txt ../..
> cd $(VENV_V100)/src/cell-detector/cell_detector/bin/
> rm -rf build
> $(VENV_V100)/bin/python setup.py build develop
> cd $(VENV_V100)/src/cell-detector/cell_detector/bin/apex/
> rm -rf build
> $(VENV_V100)/bin/python setup.py install --cuda_ext --cpp_ext

## titan
venv-titan: $(VENV_TITAN)/bin/python
$(VENV_TITAN)/bin/python:
> test -d $(VENV_TITAN) || $(CONDA) create -y -p $(VENV_TITAN) python=$(PYTHON_VERSION)
> $(VENV_TITAN)/bin/python -m pip install -U pip

install-titan: venv-titan
> $(VENV_TITAN)/bin/python -m pip install --src $(VENV_TITAN)/src -U -r requirements.cu101.titan.txt ../..
> cd $(VENV_TITAN)/src/cell-detector/cell_detector/bin/
> rm -rf build
> $(VENV_TITAN)/bin/python setup.py build develop
> cd $(VENV_TITAN)/src/cell-detector/cell_detector/bin/apex/
> rm -rf build
> $(VENV_TITAN)/bin/python setup.py install --cuda_ext --cpp_ext


clean:
> rm -fr venv
> rm -fr build/
> rm -fr dist/
> rm -fr .eggs/
> find . -name '*.egg-info' -exec rm -fr {} +
> find . -name '*.egg' -exec rm -f {} +
> find . -name '*.pyc' -exec rm -f {} +
> find . -name '*.pyo' -exec rm -f {} +
> find . -name '*~' -exec rm -f {} +
> find . -name '__pycache__' -exec rm -fr {} +
> rm -fr .tox/
> rm -fr .coverage
> rm -fr coverage.xml
> rm -fr htmlcov/
> rm -fr .pytest_cache