.. _cog-experiments:

COG Dataset and Model Implementation and Reproduction
======================================================
`@author: Emre Sevgen`

This note will explain how to partially reproduce the results of COG model on the COG dataset with our implementation.

Please refer to the paper_ for a full description of the experiments and the analysis of the results.

.. admonition:: Abstract

     A vexing problem in artificial intelligence is reasoning about events that occur in complex, \
     changing visual stimuli such as in video analysis or game play. Inspired by a rich tradition of\
     visual reasoning and memory in cognitive psychology and neuroscience, we developed an artificial, \
     configurable visual question and answer dataset (COG) to parallel experiments in humans and animals. \
     COG is much simpler than the general problem of video analysis, yet it addresses many of the \
     problems relating to visual and logical reasoning and memory -- problems that remain challenging for \
     modern deep learning architectures. We additionally propose a deep learning architecture that performs \
     competitively on other diagnostic VQA datasets (i.e. CLEVR) as well as easy settings of the COG dataset. \
     However, several settings of COG result in datasets that are progressively more challenging to learn. \
     After training, the network can zero-shot generalize to many new tasks. Preliminary analyses of the \
     network architectures trained on COG demonstrate that the network accomplishes the task in a manner \
     interpretable to humans. 


.. _paper: https://arxiv.org/abs/1803.06092

In this note, we will go through the following experiments for the COG model:

    - Training and testing on the COG dataset on canonical difficulty


If datasets aren't already downloaded, MI-Prometheus will download and unzip them for you at '~/data'.

Training the COG model on the COG dataset
------------------------------------------

The first step is to ensure that you have Mi-Prometheus up and running. Please refer to the :ref:`installation` note,
and do no hesitate to read the :ref:`explained` section, which provides in-depth information about Mi-Prometheus.

The first experiment is to train :py:class:`miprometheus.models.cog.CogModel` 
on :py:class:`miprometheus.problems.COG`.

Though data can be pre-generated, we will use the provided canonical dataset. This dataset includes a training set, 
a validation set and a test set.

The training set of COG contains 227,280 samples per task family, across 44 task families for a total of 10,000,320 samples.
By default, the canonical dataset will be placed in '~/data/cog/data_4_3_1/'

The configuration file is provided in 'mi-prometheus/configs/cog/cog_cog.yaml'

This configuration file contains all the parameters for training & validation.

Simply run

    >>> mip-online-trainer --c cog_cog.yaml --tensorboard 0 (--gpu)

The first option points to the configuration file.
The second option will log statistics using a Tensorboard writer. This will allow us to visualize the models convergence plots.
The last option, if passed, will enable training on the gpu if one is available.

.. note::

    Training on the entire dataset will take ~ 60h on a Titan X GPU.

The :py:class:`miprometheus.workers.OnlineTrainer` (called by ``mip-online-trainer``) wil create a main
experiments folder, named `experiments/<timestamp>` which will contain the statistics.

Testing the trained models on the COG test dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be completed.


Collecting the results
----------------------

To be completed.
