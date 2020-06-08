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
PYTHON_VERSION=python3.7

VENV_NAME := venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=$(VENV_NAME)/bin/python3

.PHONY: clean install

venv: $(PYTHON)

$(PYTHON):
> test -d $(VENV_NAME) || virtualenv -p $(PYTHON_VERSION) $(VENV_NAME)
> $(PYTHON) -m pip install -U pip
> touch $(VENV_NAME)/bin/activate

install: venv
> $(PYTHON) -m pip install -e .[all]

test: venv
> $(VENV_ACTIVATE) && $(PYTHON) -m pytest -e --cov-report xml --cov-report term

clean:
> rm -fr $(VENV_NAME)
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

bumpversion-major: venv
> $(VENV_ACTIVATE) && bump2version major --allow-dirty

bumpversion-minor: venv
> $(VENV_ACTIVATE) && bump2version minor --allow-dirty

bumpversion-patch: venv
> $(VENV_ACTIVATE) && bump2version patch --allow-dirty

bumpversion-dev: venv
> $(VENV_ACTIVATE) && bump2version devbuild --allow-dirty

bumpversion-release: venv
> $(VENV_ACTIVATE) && bump2version release --allow-dirty