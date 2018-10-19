
# Machine Intelligence: Prometheus

#### Bringing (Py)Torch To Mankind

|   |  |
| --- | --- |
| Version | 0.2 |
| Authors | Tomasz Kornuta, Vincent Marois, Ryan L. McAvoy, Younes Bouhadjar, Alexis Asseman, Vincent Albouy, T.S. Jayram, Ahmet S. Ozcan |
| Web site | https://github.com/IBM/mi-prometheus |
| Documentation | http://mi-prometheus.readthedocs.io/ |
| Copyright | This document has been placed in the public domain. |
| License | Mi-Prometheus is released under the Apache 2.0 License. |

[![GitHub license](https://img.shields.io/github/license/IBM/mi-prometheus.svg?&style=flat-square)](https://github.com/IBM/mi-prometheus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mi-prometheus/badge/?version=latest&style=flat-square)](https://mi-prometheus.readthedocs.io/en/latest/?badge=latest)
## Description

MI-Prometheus (Machine Intelligence – Prometheus), an open-source framework built on top of PyTorch, enabling rapid development and comparison of diverse neural network-based models. In MI-Prometheus training and testing mechanisms are no longer pinned to a specific model or problem, whereas build-in mechanisms for configuration management facilitate running experiments combining different models with problems.

A project of the Machine Intelligence team, IBM Research, Almaden.


## Core ideas

   * Problem: a dataset or a data generator, returning a batch of inputs and ground truth labels used for a model training/validation/test,
   * Model: a trainable model (i.e. a neural network),
   * Worker: a specialized application that instantiates the Problem \& Model objects and controls the interactions between them.
   * Configuration file(s): YAML file(s) containing the parameters of the Problem, Model and training procedure (e.g. terminal conditions, curriculum learning parameters),
   * Experiment: a single run (training or test) of a given Model on a given Problem, using a specific Worker and Configuration file(s).

## Core features

   * A configuration management relying on (optionally nested) human-readable YAML files,
   * Standardization of the interfaces of the components needed in a typical deep learning system: problems, models architectures, training/test procedures etc.,
   * Reusable scripts unifying the training & test procedures, enabling reproducible experiments, 
   * Automated tools for collecting statistics and logging the results of the experiments,
   * A set of scripts for running a number ("grid") of experiments on collections of CPUs/GPUs,
   * A collection of diverse problems, currently covering most of the actively explored domains,
   * A collection of (often state-of-the-art) models,
   * A set of tools to analyze the models during training and test (displaying model statistics and graph, dynamic visualizations, export of data to TensorBoard).

## Dependencies

   * PyTorch (v. 0.4)
   * MatPlotLib
   * TorchVision
   * TensorBoardX
   * Torchtext
   * Pyyaml
   * Sphinx
   * Sphinx_rtd_theme
   * Progressbar2
   * NLTK
   * H5PY
   * Pandas
   * Pillow
   * Six
   * PyQT
   

### Installation of the dependencies/required tools

PyTorch is the main library used by MI-Prometheus for tensors computations.
Please refer to the [official installation guide for PyTorch](https://github.com/pytorch/pytorch#installation) to install it.
We do not support PyTorch >= v0.4.1 (especially the v1.0 preview), but intend to.

For the other dependencies, we mostly use [conda]() as the packages & virtual environment manager.
We recommend using it to install the other libraries which MI-Prometheus is using.

Installing requirements for MI-Prometheus (tested on Ubuntu 16.14):
    
    # not available in conda
    pip install torchtext tensorboardX
    
    conda install matplotlib pyyaml ffmpeg sphinx sphinx_rtd_theme tqdm progressbar2 nltk h5py pandas pillow six pyqt -y

A `setup.py` script should be coming soon.
## Main workers

   * Offline Trainer - A traditional trainer, epoch-based and well-suited for traditional supervised training.

```console
foo@bar:~$ python workers/offline_trainer.py --h
usage: offline_trainer.py [-h] [--config CONFIG] [--model MODEL] [--gpu]
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
foo@bar:~$ python workers/online_trainer.py --h
usage: online_trainer.py [-h] [--config CONFIG] [--model MODEL] [--gpu]
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
foo@bar:~$ python workers/tester.py --h
usage: tester.py [-h] [--config CONFIG] [--model MODEL] [--gpu]
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

## Grid workers

Grid Workers manage several experiments ("_grids_") by reusing the base workers, such as OfflineTrainer \& Tester.
There are 3 types of Grid Workers:

- Grid Trainers, which span several trainings in parallel. Two versions are available: One for CPU cores (GridTrainerCPU) and one for GPUs (CUDA) (GridTrainerGPU),
- Grid Testers, which test several trained models in parallel. The same two versions are available: GridTesterCPU & GridTesterGPU,
- GridAnalyzer,which summarizes the results of several trainings & tests into one csv file.

 * Grid Trainer(s):

```console
foo@bar:~$ python workers/grid_trainer_cpu.py --h
usage: grid_trainer_cpu.py [-h] [--outdir OUTDIR] [--savetag SAVETAG]
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
foo@bar:~$ python workers/grid_tester_cpu.py --h
usage: grid_tester_cpu.py [-h] [--outdir OUTDIR] [--savetag SAVETAG]
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


## Documentation

Documentation is created using `Sphinx`. In order to generate it, you can run the following command:

    /mi-prometheus/scripts/docgen.sh

This script requires that the `Python` packages Sphinx & sphinx_rtd_theme are installed in the environment.
You should also ensure that the dependencies of MI-Prometheus are also present, as Sphinx imports the packages & modules to pull the docstrings.

## Maintainers

* Tomasz Kornuta (tkornut@us.ibm.com)
* Vincent Marois (vincent.marois@protonmail.com)
* Ryan L. McAvoy
* Younes Bouhadjar (younes.bouhadjy@gmail.com)
* Alexis Asseman
* Vincent Albouy
* T.S. Jayram (jayram@us.ibm.com)
* Ahmet S. Ozcan

