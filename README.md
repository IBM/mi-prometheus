
# Machine Intelligence: Prometheus

#### Bringing (Py)Torch To Mankind


[![GitHub license](https://img.shields.io/github/license/IBM/mi-prometheus.svg)](https://github.com/IBM/mi-prometheus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mi-prometheus/badge/?version=latest)](https://mi-prometheus.readthedocs.io/en/latest/?badge=latest)
[![GitHub version](https://badge.fury.io/gh/IBM%2Fmi-prometheus.svg)](https://badge.fury.io/gh/IBM%2Fmi-prometheus)


- [Description](#description)
- [Installation](#installation)
- [Core ideas](#core-ideas)
- [Core features](#core-features)
    - [Main Workers](#main-workers)
    - [Grid Workers](#grid-workers)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [The Team](#the-team)


## Description

MI-Prometheus (Machine Intelligence - Prometheus), an open-source framework aiming at _accelerating Machine Learning Research_, by fostering the rapid development of diverse neural network-based models and facilitating their comparison. 
In its core, to _accelerate the computations_ on their own, MI-Prometheus relies on PyTorch and extensively uses its mechanisms for the distribution of computations on CPUs/GPUs.

In MI-Prometheus, the training & testing mechanisms are no longer pinned to a specific model or problem, and built-in mechanisms for easy configuration management & statistics collection facilitate running experiments combining different models with problems.

A project of the Machine Intelligence team, IBM Research, Almaden.


### Installation

PyTorch is the main library used by MI-Prometheus for tensors computations.
Please refer to the [official installation guide for PyTorch](https://github.com/pytorch/pytorch#installation) to install it.
We currently do not officially support PyTorch >= v0.4.1 (especially the v1.0 preview), but intend to in the near future.

To install MI-Prometheus, you can use the `setup.py` script with the following command:

    python setup.py install

We will upload MI-prometheus to [PyPI](https://pypi.org/) in the near future.

The dependencies of MI-prometheus are: 

   * pytorch (v. 0.4.0)
   * numpy
   * torchvision (v. 0.2.0)
   * torchtext
   * tensorboardx
   * matplotlib 
   * PyYAML
   * tqdm
   * nltk
   * h5py
   * six
   * pyqt5 (v. 5.10.1)


## Core ideas

   * **Problem**: A dataset or a data generator, returning a batch of inputs and ground truth labels used for a model training/validation/test,
   * **Model**: A trainable model (i.e. a neural network),
   * **Worker**: A specialized application that instantiates the Problem \& Model objects and controls the interactions between them,
   * **Configuration file(s)**: YAML file(s) containing the parameters of the Problem, Model and training procedure (e.g. terminal conditions, random seeds), divided into several sections,
   * **Experiment**: A single run (training & validation or test) of a given Model on a given Problem, using a specific Worker and Configuration file(s). Such an Experiment also collects and logs diverse statistics during its execution.

## Core features

   * A configuration management relying on (optionally nested) human-readable YAML files,
   * Reusable scripts unifying the training & test procedures, enabling reproducible experiments, 
   * Automated tools for collecting statistics and logging the results of the experiments,
   * A set of scripts for running a number ("grid") of experiments on collections of CPUs/GPUs,
   * A collection of diverse problems, currently covering most of the actively explored domains,
   * A collection of (often state-of-the-art) models,
   * A set of tools to analyze the models during training and test (displaying model statistics and graph, dynamic visualizations, export of data to TensorBoard).
   



### Base workers

The base workers are the main way you will use MI-Prometheus. They are parameterizable, OOP-designed scripts which will execute a specific task related to the supervised training or test of a Model on a Problem, following a Configuration.

   * Offline Trainer - A traditional trainer, epoch-based and well-suited for traditional supervised training.

```console
foo@bar:~$ mip-offline-trainer --h
usage: mip-offline-trainer [-h] [--config CONFIG] [--model MODEL] [--gpu]
                           [--outdir OUTDIR] [--savetag SAVETAG]
                           [--ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                           [--li LOGGING_INTERVAL] [--agree]
                           [--tensorboard {0,1,2}] [--visualize {-1,0,1,2,3}]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Name of the configuration file(s) to be loaded. If specifying more than one file, they must be separated with coma ",".
  --model MODEL         Path to the file containing the saved parameters of the model to load (model checkpoint, should end with a .pt extension.)
  --gpu                 The current worker will move the computations on GPU devices, if available in the system. (Default: False)
  --outdir OUTDIR       Path to the output directory where the experiment(s) folders will be stored. (DEFAULT: ./experiments)
  --savetag SAVETAG     Tag for the save directory
  --ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level. (Default: INFO)
  --li LOGGING_INTERVAL
                        Statistics logging interval. Will impact logging to the logger and exporting to TensorBoard. Writing to the csv file is not impacted (interval of 1). (Default: 100, i.e. logs every 100 episodes).
  --agree               Request user confirmation just after loading the settings, before starting training  (Default: False)
  --tensorboard {0,1,2}
                        If present, enable logging to TensorBoard. Available log levels:
                        0: Log the collected statistics.
                        1: Add the histograms of the model's biases & weights (Warning: Slow).
                        2: Add the histograms of the model's biases & weights gradients (Warning: Even slower).
  --visualize {-1,0,1,2,3}
                        Activate dynamic visualization (Warning: will require user interaction):
                        -1: disabled (DEFAULT)
                        0: Only during training episodes.
                        1: During both training and validation episodes.
                        2: Only during validation episodes.
                        3: Only during the last validation, after the training is completed.

```

   * Online Trainer - A different type of trainer, more flexible and well-suited for problems generating samples _on-the-fly_.

```console
foo@bar:~$ mip-online-trainer --h
usage: mip-online-trainer [-h] [--config CONFIG] [--model MODEL] [--gpu]
                          [--outdir OUTDIR] [--savetag SAVETAG]
                          [--ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                          [--li LOGGING_INTERVAL] [--agree]
                          [--tensorboard {0,1,2}] [--visualize {-1,0,1,2,3}]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Name of the configuration file(s) to be loaded. If specifying more than one file, they must be separated with coma ",".
  --model MODEL         Path to the file containing the saved parameters of the model to load (model checkpoint, should end with a .pt extension.)
  --gpu                 The current worker will move the computations on GPU devices, if available in the system. (Default: False)
  --outdir OUTDIR       Path to the output directory where the experiment(s) folders will be stored. (DEFAULT: ./experiments)
  --savetag SAVETAG     Tag for the save directory
  --ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level. (Default: INFO)
  --li LOGGING_INTERVAL
                        Statistics logging interval. Will impact logging to the logger and exporting to TensorBoard. Writing to the csv file is not impacted (interval of 1). (Default: 100, i.e. logs every 100 episodes).
  --agree               Request user confirmation just after loading the settings, before starting training  (Default: False)
  --tensorboard {0,1,2}
                        If present, enable logging to TensorBoard. Available log levels:
                        0: Log the collected statistics.
                        1: Add the histograms of the model's biases & weights (Warning: Slow).
                        2: Add the histograms of the model's biases & weights gradients (Warning: Even slower).
  --visualize {-1,0,1,2,3}
                        Activate dynamic visualization (Warning: will require user interaction):
                        -1: disabled (DEFAULT)
                        0: Only during training episodes.
                        1: During both training and validation episodes.
                        2: Only during validation episodes.
                        3: Only during the last validation, after the training is completed.


```
   
   * Tester - A worker which loads a pretrained model and tests it on a given problem.

```console
foo@bar:~$ mip-tester --h
usage: mip-tester [-h] [--config CONFIG] [--model MODEL] [--gpu]
                  [--outdir OUTDIR] [--savetag SAVETAG]
                  [--ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                  [--li LOGGING_INTERVAL] [--agree] [--visualize]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Name of the configuration file(s) to be loaded. If specifying more than one file, they must be separated with coma ",".
  --model MODEL         Path to the file containing the saved parameters of the model to load (model checkpoint, should end with a .pt extension.)
  --gpu                 The current worker will move the computations on GPU devices, if available in the system. (Default: False)
  --outdir OUTDIR       Path to the output directory where the experiment(s) folders will be stored. (DEFAULT: ./experiments)
  --savetag SAVETAG     Tag for the save directory
  --ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level. (Default: INFO)
  --li LOGGING_INTERVAL
                        Statistics logging interval. Will impact logging to the logger and exporting to TensorBoard. Writing to the csv file is not impacted (interval of 1). (Default: 100, i.e. logs every 100 episodes).
  --agree               Request user confirmation just after loading the settings, before starting training  (Default: False)
  --visualize           Activate dynamic visualization

```

### Grid workers

Grid Workers manage several experiments ("_grids_") by reusing the base workers, such as OfflineTrainer \& Tester.
There are 3 types of Grid Workers:

- mip-grid-trainer-*, which span several trainings in parallel. Two versions are available: One for CPU cores (`GridTrainerCPU`) and one for GPUs (CUDA) (`GridTrainerGPU`),
- mip-grid-tester-*, which test several trained models in parallel. The same two versions are available: `GridTesterCPU` & `GridTesterGPU`,
- mip-grid-analyzer, which summarizes the results of several trainings & tests into one csv file.

 * Grid Trainer(s):

```console
foo@bar:~$ mip-grid-trainer-cpu --h
usage: mip-grid-trainer-cpu [-h] [--outdir OUTDIR] [--savetag SAVETAG]
                           [--ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                           [--li LOGGING_INTERVAL] [--agree] [--config CONFIG]
                           [--online_trainer] [--tensorboard {0,1,2}]

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Path to the global output directory where the experiments folders will be / are stored. Affects all grid experiments. (DEFAULT: ./experiments)
  --savetag SAVETAG     Additional tag for the global output directory.
  --ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level for the experiments. (Default: INFO)
  --li LOGGING_INTERVAL
                        Statistics logging interval. Will impact logging to the logger and exporting to TensorBoard for the experiments. Do not affect the grid worker. Writing to the csv file is not impacted (interval of 1). (Default: 100, i.e. logs every 100 episodes).
  --agree               Request user confirmation before starting the grid experiment.  (Default: False)
  --config CONFIG       Name of the configuration file(s) to be loaded. If specifying more than one file, they must be separated with coma ",".
  --online_trainer      Select the OnLineTrainer instead of the default OffLineTrainer.
  --tensorboard {0,1,2}
                        If present, enable logging to TensorBoard. Available log levels:
                        0: Log the collected statistics.
                        1: Add the histograms of the model's biases & weights (Warning: Slow).
                        2: Add the histograms of the model's biases & weights gradients (Warning: Even slower).
```

 * Grid Tester(s):

```console
foo@bar:~$ mip-grid-tester-cpu --h
usage: mip-grid-tester-cpu [-h] [--outdir OUTDIR] [--savetag SAVETAG]
                          [--ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                          [--li LOGGING_INTERVAL] [--agree] [--n NUM_TESTS]

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Path to the global output directory where the experiments folders will be / are stored. Affects all grid experiments. (DEFAULT: ./experiments)
  --savetag SAVETAG     Additional tag for the global output directory.
  --ll {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level for the experiments. (Default: INFO)
  --li LOGGING_INTERVAL
                        Statistics logging interval. Will impact logging to the logger and exporting to TensorBoard for the experiments. Do not affect the grid worker. Writing to the csv file is not impacted (interval of 1). (Default: 100, i.e. logs every 100 episodes).
  --agree               Request user confirmation before starting the grid experiment.  (Default: False)
  --n NUM_TESTS         Number of test experiments to run for each model.
```

 * Grid Analyzer: Similar options.

**NOTES**: 
* We primarily test MI-Prometheus on CUDA devices, as they are our main hardware setup and PyTorch mainly supports CUDA as a backend.
* We currently are using a utility called [cuda-gpupick](https://github.com/aasseman/cuda-gpupick) to  pick unused CUDA devices (in a topology-aware fashion) for the GPU versions of the Grid Workers.
While this utility is easily installable and usable, we understand that this is a supplementary constraint, which we will work on to relax it.

   
## Documentation

Documentation is created using `Sphinx`, and is available on [readthedocs.io](https://mi-prometheus.readthedocs.io/en/latest/).

## Getting Started

- [Tutorials: get you started with understanding and using MI-prometheus](): Coming soon!
- [The API Reference](https://mi-prometheus.readthedocs.io/en/latest/)

## Contributing

You are encouraged if you would like to contribute! Please use the [issues](https://github.com/IBM/mi-prometheus/issues) if you want to request a new feature or a fix, so that we can discuss it first.

## The Team

* Tomasz Kornuta (tkornut@us.ibm.com)
* Vincent Marois (vincent.marois@protonmail.com)
* Ryan L. McAvoy
* Younes Bouhadjar (younes.bouhadjy@gmail.com)
* Alexis Asseman
* Vincent Albouy
* T.S. Jayram (jayram@us.ibm.com)
* Ahmet S. Ozcan

