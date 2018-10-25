MI-Prometheus detailed architecture
=====================================

This page dives deep into MI-Prometheus and its inner workings.

Core concepts
---------------

When training a model, people write programs which typically follow a similar pattern:

    - Loading the data samples & instantiating the model, 
    - Feeding the model batches of image-label pairs, which are passed through the model forward pass,
    - Computing the loss as a difference between the predicted labels and the ground truth labels, 
    - This error is propagated backwards using backpropagation,
    - Updating the model parameters using an optimizer.
    

During each iteration, the program also needs to collect some statistics (such as the
training / validation loss & accuracy) and save the weights of the resulting model into a file.


This typical workflow led us to the formalization of the core concepts of the framework:

    - **Problem**: a dataset or a data generator, returning a batch of inputs and ground truth labels used for a model training/validation/test,
    - **Model**: a trainable model (i.e. a neural network),
    - **Worker**: a specialized application that instantiates the Problem & Model objects and controls the interactions between them.
    - **Configuration file(s)**: YAML file(s) containing the parameters of the Problem, Model and training procedure,
    - **Experiment**: a single run (training or test) of a given Model on a given Problem, using a specific Worker and Configuration file(s).


.. figure:: ../img/core_concepts.png
   :scale: 50 %
   :alt: The 5 core concepts of Mi-Prometheus
   :align: center

   The 5 core concepts of Mi-Prometheus. Dotted elements indicate optional inputs/outputs/dataflows.

Architecture
---------------

From an architectural point of view, MI-Prometheus can be seen as four stacked layers of interconnected modules.

	- The lowest layer is formed by the external libraries that MI-Prometheus relies on, primarily PyTorch, NumPy and CUDA. Additionally, our basic workers rely on TensorBoardX, enabling the export of collected statistics, models and their parameters (weights, gradients) to TensorBoard. Optionally, some models and problems might depend on other external libraries. For instance, the framework currently incorporates problems and models from PyTorch’s wrapper to the TorchVision package.
	- The second layer includes all the utilities that we have developed internally, such as the Parameter Registry (a singleton offering access to the registry of parameters), the Application State (another singleton representing the current state of application, e.g. whether the computations should be done on GPUs or not), factories used by the workers for instantiating the problem and model classes (indicated by the configuration file and loaded from the corresponding file). Additionally, this layer contains several tools, which are useful during an experiment run, such as logging facilities or statistics collectors (accessible by both the Problem and the Model).
	- Next, the Components layer contains the models, problems and workers, i.e. the three major components required for the execution of one experiment. The problem and model classes are organized following specific hierarchies, using inheritance to facilitate their further extensions.
	- Finally, the Experiment layer includes the configuration files, along with all the required inputs (such as the files containing the dataset, the files containing the saved model checkpoints with the weights to be loaded etc.) and outputs (logs from the experiment run, CSV files gathering the collected statistics, files containing the checkpoints of the best obtained model).


.. figure:: ../img/layers.png
   :scale: 50 %
   :alt: Mi-Prometheus is constituted of 4 main inter-connected layers.
   :align: center

   From an architectural point of view, MI-Prometheus can be seen as four stacked layers of interconnected modules.


.. See http://docutils.sourceforge.net/docs/ref/rst/directives.html for a breakdown of the options