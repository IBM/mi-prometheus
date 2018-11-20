.. _vigil-experiments:

VIGIL Workshop experiments
===========================
`@author: Vincent Marois`

This note will explain how to reproduce the VIGIL experiments for MAC & S-MAC.

<``Will have to write some more words here to better introduce the experiments``>

The first thing to ensure is that you have downloaded & unzipped the CLEVR_ (18GB) & `CLEVR-CoGenT`_ (24GB) datasets.
The following will assume that you have placed the folders (which should be `CLEVR_v1.0` & `CLEVR_CoGenT_v1.0`
after unzipping) in `~/data/`.

Training MAC & S-MAC on CLEVR & CoGenT
------------------------------------------

The first step is to ensure that you have Mi-Prometheus up and running. Please refer to the :ref:`installation` note,
and do no hesitate to read the :ref:`explained` section, which provides in-depth information about Mi-Prometheus.

The first experiment is to trained :py:class:`miprometheus.models.mac.MACNetwork` and
:py:class:`miprometheus.models.s_mac.sMacNetwork` on CLEVR & CLEVR-CoGenT A.

Given that the ground truth answers for the test sets of CLEVR & CoGenT (both conditions) are not distributed by the authors,
we will use the original validation sets as test sets, and thus split the original training sets into a train set (90%)
and a validation set (10%).

The :py:class:`miprometheus.helpers.IndexSplitter` is designed just for that. We simply need to provide the length
of the dataset we need to split and how many samples will be contained in the first split.

The training set of CLEVR contains 699 989 samples, and we keep 629 990 for training.
The training set of CoGenT-A contains 699 960 samples, and we keep 629 964 for training.

Just run

    >>> mip-index-splitter --l 699989 --s 629990

This command will generate 2 files, `split_a.txt` and `split_b.txt` which contains the samples indices to index
90% of the training set and 10% of the training set respectively.

Rename these files to `vigil_clevr_train_set_indices.txt` and `vigil_clevr_val_set_indices.txt` respectively, and place
them in `~/data/CLEVR_v1.0/`.

Do the same operation for CoGenT:

    >>> mip-index-splitter --l 699960 --s 629964

Rename the files to `vigil_cogent_train_set_indices.txt` and `vigil_cogent_val_set_indices.txt` respectively, and place
them in `~/data/CLEVR_CoGenT_v1.0/`.

Here are the ones we used for reference:

:download:`CLEVR-train <vigil/vigil_clevr_train_set_indices.txt>`
:download:`CLEVR-val <vigil/vigil_clevr_val_set_indices.txt>`
:download:`CoGenT-train <vigil/vigil_cogent_train_set_indices.txt>`
:download:`CoGenT-val <vigil/vigil_cogent_val_set_indices.txt>`


A grid configuration file is available to run the 4 initial training experiments:

=======  =======
 Model   Dataset
=======  =======
  MAC     CLEVR
 S-MAC    CLEVR
  MAC     CoGenT
 S-MAC    CoGenT
=======  =======


:download:`Grid configuration file <../../../configs/mac/mac_smac_initial_training.yaml>`

Simply run

    >>> mip-grid-trainer-gpu --c mac_smac_initial_training.yaml --savetag vigil_workshop --tensorboard 0

The first option points to the grid configuration file.
The second option indicates an additional tag for the experiments folder.
The last option will log statistics using a Tensorboard writer. This will allow us to visualize the models convergence plots.

.. note::

    Training for 20 epochs will take ~ 24h on a GPU (one GPU per experiment).



.. _CLEVR: https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
.. _CLEVR-CoGenT: https://s3-us-west-1.amazonaws.com/clevr/CLEVR_CoGenT_v1.0.zip
