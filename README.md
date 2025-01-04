# Scientific Quantum Utilities and Investigations Breakthough (SQUIB)

## Description

This repository is a prototyping space for implementations of quantum algorithms

## Installation

To install this module using pipenv, run the following:

```
pipenv install
```

To install development tools, add the `--dev` option.

To install in your python environment manually, use the following commands:

```
python3 setup.py sdist
python3 setup.py install
```

## Modules

### QKNN

The Quantum k-nearest neighbors module contains all the code necessary to run the k-nearest neighbors algorithm on sets of vectors using Qiskit's quantum computing simulators. 

### quclidean

The quclidean module contains circuit building functions that construct the circuits necessary to perform simultaneous Euclidean distance calculations between two sets of real-valued vectors.  There are a number of functions available to suit the method of circuit construction desired.

## Scripts

### experiments.py

The script found at `notebooks/experiments.py` automates the building and running of QKNN experiments for a number of UC Irvine Machine Learning repository datasets. To run this script, you must have access to the internet to download the necessary datasets.

#### Usage

Running `experiments.py` from the command looks like:

```
python3 experiments.py [options]
```

#### Options

|Name|Flag|Description|
|----|----|-----------|
|dataset|-d|The name of the UCI dataset|
|output_directory|-o|The directory in which to save the results|
|log_directory|-l|The directory in which to save log files to|
|k|-k|The k-value for k-nearest neighbors|
|backend|-b|The name of the backend to use|
|shots|-s|The shots for which to run the simulator|
|seed|--seed|The seed to use for random generators|

**dataset options**
- iris
- transfusion
- vertebral
- ecoli
- glass
- breast_cancer

**backend options**
- aersimulator
- statevectorsimulator