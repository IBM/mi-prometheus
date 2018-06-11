
# Machine Intelligence: Prometheus

#### Bringing Torch To Mankind

## Description

A project of the Machine Intelligence team, focusing on enabling the applications of machine learning to be easier to compare and reproduce.


## Core ideas

   * model - class representing an actual (trainable) model (class derived from torch.nn.Module).
   * problem - class returning generator producing data used for training/validation/testing of models (generalization of Data Generators and Datasets).
   * configuration file - file containing set of parameters useful for training/testing of a given model on a given problem
   * experiment - a single run of training or testing procedure 
   * worker - application that reads configuration file, creates associated problem and model objects and runs experiment 

## Core features

   * Configuration management based on yaml files
   * Automatization of training/validation/testing pipelines
   * Integration with TensorBoard
   * Advanced Visualization with MatPlotLib

## Dependencies

   * PyTorch (v. 0.4)
   * MatPlotLib
   * TorchVision
   * TensorBoardX

### Installation of the dependencies/required tools

On Linux (Ubuntu 14.04 on MacBook Pro, without CUDA): 

    conda install pytorch-cpu torchvision-cpu -c pytorch
    conda install -c conda-forge tensorboardx 


## Main workers

   * train - application for model training.

```console
foo@bar:~$ python train.py --h
usage: train.py [-h] [--agree] [--config CONFIG] [--tensorboard {0,1,2}]
                [--lf LOGGING_FREQUENCY]
                [--log {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]
                [--visualize {0,1,2,3}]

optional arguments:
  -h, --help            show this help message and exit
  --agree               Request user confirmation just after loading the settings, before starting training  (Default: False)
  --config CONFIG       Name of the configuration file to be loaded
  --tensorboard {0,1,2}
                        If present, log to TensorBoard. Log levels:
                        0: Just log the loss, accuracy, and seq_len
                        1: Add histograms of biases and weights (Warning: slow)
                        2: Add histograms of biases and weights gradients (Warning: even slower)
  --lf LOGGING_FREQUENCY
                        TensorBoard logging frequency (Default: 100, i.e. logs every 100 episodes)
  --log {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
                        Log level. (Default: INFO)
  --visualize {0,1,2,3}
                        Activate dynamic visualization:
                        0: Only during training
                        1: During both training and validation
                        2: Only during validation
                        3: Only during last validation, after training is completed
```

   * train - application that loads the pretrained models and tests them on a given problem.

```console
foo@bar:~$ python train.py --h
usage: test.py [-h] [--model MODEL] [--visualize]
               [--log {critical,error,warning,info,debug,notset}]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to and name of the file containing the saved
                        parameters of the model (model checkpoint)
  --visualize           Activate dynamic visualization
  --log {critical,error,warning,info,debug,notset}
                        Log level. Default is INFO.
```


# How to Run the code: 
   * Training: ```python train.py --v 1 --c configs/dwm/serial_recall.yaml```

   * Testing:  ``` python test.py --m path_to_model --v 1```


## Documentation

In order to generate a "living" documentation of the code please run Sphinx (TODO)

## Maintainer

Tomasz Kornuta (tkornut@us.ibm.com)
