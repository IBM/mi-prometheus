.. _vigil-experiments:

VIGIL Workshop experiments
===========================
`@author: Vincent Marois`

This note will explain how to reproduce the VIGIL experiments for MAC & S-MAC.

Please refer to the paper_ for a full description of the experiments and the analysis of the results.

.. admonition:: Abstract

    We introduce a variant of the MAC model (Hudson and Manning, ICLR 2018) \
    with a simplified set of equations that achieves comparable accuracy, while train-\
    ing faster. We evaluate both models on CLEVR and CoGenT, and show that, trans-\
    fer learning with fine-tuning results in a 15 point increase in accuracy, matching \
    the state of the art. Finally, in contrast, we demonstrate that improper fine-tuning \
    can actually reduce a model’s accuracy as well.


.. _paper: https://arxiv.org/abs/1811.06529

In this note, we will go through the following experiments, both for MAC & S-MAC:

    - Initial training on CLEVR,
        - Test on CLEVR, CoGenT-A & CoGenT-B
    - Initial training on CoGenT-A,
        - Test on CoGenT-A & CoGenT-B
    - Finetuning the CoGenT-A- & CLEVR-trained models on CoGenT-B,
        - Test on CoGenT-A & CoGenT-B
    - Finetuning the CLEVR-trained models on CoGenT-A
        - Test on CoGenT-A & CoGenT-B


The first thing to ensure is that you have downloaded & unzipped the CLEVR_ (18GB) & `CLEVR-CoGenT`_ (24GB) datasets.
The following will assume that you have placed the folders (which should be named `CLEVR_v1.0/` & `CLEVR_CoGenT_v1.0/`
after unzipping) in `~/data/`.

.. _CLEVR: https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
.. _CLEVR-CoGenT: https://s3-us-west-1.amazonaws.com/clevr/CLEVR_CoGenT_v1.0.zip

Training MAC & S-MAC on CLEVR & CoGenT
------------------------------------------

The first step is to ensure that you have Mi-Prometheus up and running. Please refer to the :ref:`installation` note,
and do no hesitate to read the :ref:`explained` section, which provides in-depth information about Mi-Prometheus.

The first experiment is to train :py:class:`miprometheus.models.mac.MACNetwork` and
:py:class:`miprometheus.models.s_mac.sMacNetwork` on CLEVR & CLEVR-CoGenT A.

Given that the ground truth answers for the test sets of CLEVR & CoGenT (both conditions) are not distributed by the authors,
we will use the original validation sets as test sets, and thus split the original training sets into a train set (90%)
and a validation set (10%).

The :py:class:`miprometheus.helpers.IndexSplitter` is designed just for that. We simply need to provide the length
of the dataset we need to split and how many samples will be contained in the first split.

The training set of CLEVR contains 699,989 samples, and we keep 629,990 for training.
The training set of CoGenT-A contains 699,960 samples, and we keep 629,964 for training.

Just run

    >>> mip-index-splitter --l 699989 --s 629990 --o '~/data/CLEVR_v1.0/'

This command will generate 2 files, `split_a.txt` and `split_b.txt` which contains the samples indices to index
90% of the training set and 10% of the training set respectively, and place them in `~/data/CLEVR_v1.0/`.

Rename these files to `vigil_clevr_train_set_indices.txt` and `vigil_clevr_val_set_indices.txt` respectively.

Do the same operation for CoGenT:

    >>> mip-index-splitter --l 699960 --s 629964 --o '~/data/CLEVR_v1.0/'

Rename the files to `vigil_cogent_train_set_indices.txt` and `vigil_cogent_val_set_indices.txt` respectively.

Here are the ones we used for reference:

:download:`vigil_clevr_train_set_indices.txt <vigil/vigil_clevr_train_set_indices.txt>`
:download:`vigil_clevr_val_set_indices.txt <vigil/vigil_clevr_val_set_indices.txt>`
:download:`vigil_cogent_train_set_indices.txt <vigil/vigil_cogent_train_set_indices.txt>`
:download:`vigil_cogent_val_set_indices.txt <vigil/vigil_cogent_val_set_indices.txt>`


A grid configuration file is available to run the 4 initial training experiments:

+-------+--------------+---------------------------------+
| Model | Training set |             Test sets           |
+=======+==============+=================================+
|  MAC  |    CLEVR     |    CLEVR / CoGenT-A / CoGenT-B  |
+-------+--------------+---------------------------------+
| S-MAC |    CLEVR     |    CLEVR / CoGenT-A / CoGenT-B  |
+-------+--------------+---------------------------------+
|  MAC  |  CoGenT-A    |        CoGenT-A / CoGenT-B      |
+-------+--------------+---------------------------------+
| S-MAC |  CoGenT-A    |        CoGenT-A / CoGenT-B      |
+-------+--------------+---------------------------------+

:download:`mac_smac_initial_training.yaml <../../../configs/mac/mac_smac_initial_training.yaml>`

This configuration file contains all the parameters for training & validation, as well as the multiple tests to run
on CLEVR / CoGenT-A / CoGenT-B. You can have a look at the ``multi_tests`` key in the ``testing`` section of each
specified experiment for the tests which will be run with the corresponding trained model.

Simply run

    >>> mip-grid-trainer-gpu --c mac_smac_initial_training.yaml --savetag initial_training --tensorboard 0

The first option points to the grid configuration file.
The second option indicates an additional tag for the experiments folder.
The last option will log statistics using a Tensorboard writer. This will allow us to visualize the models convergence plots.

.. note::

    Training for 20 epochs will take ~ 24h on a GPU (one GPU per experiment).

The :py:class:`miprometheus.grid_workers.GridTrainerGPU` (called by ``mip-grid-trainer-gpu``) wil create a main
experiments folder, named `experiments_<timestamp>_initial_training` which will contain subfolders for the individual
experiments (:py:class:`miprometheus.models.mac.MACNetwork` on :py:class:`miprometheus.problems.CLEVR`,
:py:class:`miprometheus.models.s_mac.sMacNetwork` on :py:class:`miprometheus.problems.CLEVR` etc.).

Testing the trained models on CLEVR / CoGenT-A / CoGenT-B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once these initial experiments are finished, we can use the :py:class:`miprometheus.grid_workers.GridTesterGPU`
to run the tests experiments which are indicated in the initial configuration file.

Simply run

    >>> mip-grid-tester-gpu --e experiments_<timestamp>_initial_training/

This will spawn a :py:class:`miprometheus.workers.Tester` for each individual experiment, which will run a test for
each set of parameters indicated in the ``multi_tests`` key in the ``testing`` section of each configuration file,
and store the results in each experiment sub-folder.


Finetuning the CoGenT-A & CLEVR trained models on CoGenT-B
-----------------------------------------------------------

The second training experiment is to finetune the CoGenT-A- & CLEVR-trained MAC & S-MAC on CoGenT-B to observe if this
increases their performance on zero-shot learning from CoGenT-A to CoGenT-B (as both CoGenT datasets contain
complementary subsets of colors/shapes combinations present in CLEVR).

We finetune these models on 30k samples of the validation set of the CoGenT-B condition and keep the complementary
samples for testing. We use the entire validation set of the CoGenT-A for testing.

The CoGenT-B validation set contains 149,991 samples. Run

    >>> mip-index-splitter --l 149991 --s 30000 --o '~/data/CLEVR_CoGenT_v1.0/'

to split the range of indices in 2. Rename the files to `vigil_cogent_finetuning_valB_indices.txt` and
`vigil_cogent_test_valB_indices.txt` respectively. You can also use ours:

:download:`vigil_cogent_finetuning_valB_indices.txt <vigil/vigil_cogent_finetuning_valB_indices.txt>`
:download:`vigil_cogent_test_valB_indices.txt <vigil/vigil_cogent_test_valB_indices.txt>`


Also, download and place in `~/data/CLEVR_CoGenT_v1.0/` the following file:

:download:`vigil_cogent_valA_full_indices.txt <vigil/vigil_cogent_valA_full_indices.txt>`

This file contains all indices of the CoGenT-A validation set samples indices and simply makes the configuration easier.

A grid configuration file is available to run these 4 experiments:

+------------------------+----------------+------------------------+
|         Model          | Finetuning set |       Test sets        |
+========================+================+========================+
|  CoGenT-A-trained MAC  |    CoGenT-B    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+
| CoGenT-A-trained S-MAC |    CoGenT-B    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+
|    CLEVR-trained MAC   |    CoGenT-B    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+
|   CLEVR-trained S-MAC  |    CoGenT-B    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+

:download:`mac_smac_cogent_b_finetuning.yaml <../../../configs/mac/mac_smac_cogent_b_finetuning.yaml>`

This configuration file contains all the information for finetuning on CoGenT-B.

.. note::

    In this file, you need to indicate the filepath to the trained models that the \
    :py:class:`miprometheus.workers.OfflineTrainer` needs to load in order to finetune it.

    In each sub-section of the ``grid_tasks`` section, there is a ``model`` section only containing a ``load`` key. \
    Here, indicate the path to the trained models, which will be as follows:

        - MAC on CLEVR: In `experiments_<timestamp>_initial_training/CLEVR/MACNetwork/`, there should be 2 timestamped \
          folder: one for MAC on CLEVR and one for MAC on CoGenT-A. Ideally, the earliest timestamp should correspond to \
          MAC on CLEVR, and the other to MAC on CoGenT-A. You can check the respective `training_configuration.yaml` \
          file in each folder to ensure this.

          The path to the trained MAC should then be: \
          `experiments_<timestamp>_initial_training/CLEVR/MACNetwork/<timestamp>/models/model_best.pt`

        - S-MAC on CLEVR: Exactly similar to MAC on CLEVR, where the path should then be: \
          `experiments_<timestamp>_initial_training/CLEVR/sMacNetwork/<timestamp>/models/model_best.pt`


    Indicate the respective paths for the 4 experiments and save the file.


You can have a look at the ``multi_tests`` key in the ``testing`` section of each specified experiment
for the tests which will be run with the corresponding trained model.

Simply run

    >>> mip-grid-trainer-gpu --c mac_smac_cogent_b_finetuning.yaml --savetag cogent_b_finetuning --tensorboard 0

This will further train the CLEVR- & CoGenT-A-trained models on CoGenT-B data for 10 epochs.

Testing the finetuned models on CoGenT-A / CoGenT-B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply run

    >>> mip-grid-tester-gpu --e experiments_<timestamp>_cogent_b_finetuning/

This will spawn a :py:class:`miprometheus.workers.Tester` for each individual experiment, which will run a test for
each set of parameters indicated in the ``multi_tests`` key in the ``testing`` section of each configuration file,
and store the results in each experiment sub-folder.


Finetuning the CLEVR-trained models on CoGenT-A
-----------------------------------------------

The last training experiment is to finetune the CLEVR-trained models on CoGenT-A to observe if this
increases their performance on CoGenT-A and/or CoGenT-B (as both CoGenT datasets contain complementary subsets of col-
ors/shapes combinations present in CLEVR).

We finetune these models on 30k samples of each validation set and keep the complementary samples for testing.
We use the entire validation set of the CoGenT-A for testing.

The CoGenT-A validation set contains 150,000 samples. Run

    >>> mip-index-splitter --l 150000 --s 30000 --o '~/data/CLEVR_CoGenT_v1.0/'

to split the range of indices in 2. Rename the files to `vigil_cogent_finetuning_valA_indices.txt` and
`vigil_cogent_test_valA_indices.txt` respectively. You can also use ours:

:download:`vigil_cogent_finetuning_valA_indices.txt <vigil/vigil_cogent_finetuning_valA_indices.txt>`
:download:`vigil_cogent_test_valA_indices.txt <vigil/vigil_cogent_test_valA_indices.txt>`


Also, download and place in `~/data/CLEVR_CoGenT_v1.0/` the following file:

:download:`vigil_cogent_valB_full_indices.txt <vigil/vigil_cogent_valB_full_indices.txt>`

This file contains all indices of the CoGenT-B validation set samples indices and simply makes the configuration easier.

A grid configuration file is available to run these 2 experiments:

+------------------------+----------------+------------------------+
|         Model          | Finetuning set |       Test sets        |
+========================+================+========================+
|    CLEVR-trained MAC   |    CoGenT-A    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+
|   CLEVR-trained S-MAC  |    CoGenT-A    |   CoGenT-A / CoGenT-B  |
+------------------------+----------------+------------------------+

:download:`mac_smac_cogent_a_finetuning.yaml <../../../configs/mac/mac_smac_cogent_a_finetuning.yaml>`

This configuration file contains all the information for finetuning on CoGenT-A.

.. note::

    Don't forget to add the path to the CLEVR-trained models for the 2 experiments in this configuration file.


You can have a look at the ``multi_tests`` key in the ``testing`` section of each specified experiment
for the tests which will be run with the corresponding trained model.

Simply run

    >>> mip-grid-trainer-gpu --c mac_smac_cogent_a_finetuning.yaml --savetag cogent_a_finetuning --tensorboard 0

This will further train the CLEVR trained models on CoGenT-A data for 10 epochs.

Testing the finetuned models on CoGenT-A / CoGenT-B
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simply run

    >>> mip-grid-tester-gpu --e experiments_<timestamp>_cogent_a_finetuning/

This will spawn a :py:class:`miprometheus.workers.Tester` for each individual experiment, which will run a test for
each set of parameters indicated in the ``multi_tests`` key in the ``testing`` section of each configuration file,
and store the results in each experiment sub-folder.


Collecting the results
----------------------

Now that we have several training, finetuning and tests experiments results, we can collect them using \
:py:class:`miprometheus.grid_workers.GridAnalyzer`.

Run

>>> mip-grid-analyzer --e experiments_<timestamp>_initial_training/
>>> mip-grid-analyzer --e experiments_<timestamp>_cogent_a_finetuning/
>>> mip-grid-analyzer --e experiments_<timestamp>_cogent_b_finetuning/

These commands should collect all results contained in the indicated main experiments folder and gather them in a
csv file stored at the root of the indicated folder.


`If you find this page useful, please refer to it with the following BibTex:`

::

    @article{marois2018transfer,
            title={On transfer learning using a MAC model variant},
            author={Marois, Vincent and Jayram, TS and Albouy, Vincent and Kornuta, Tomasz and Bouhadjar, Younes and Ozcan, Ahmet S},
            journal={arXiv preprint arXiv:1811.06529},
            year={2018}
    }

