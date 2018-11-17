MNIST image classification using the LeNet-5 model
==========================================================
`@author: Tomasz Kornuta`

This section explains how to incorporate and train an instance of `LeNet-5 model <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_ on 
the `MNIST dataset  <http://yann.lecun.com/exdb/mnist/>`_ using MI-Prometheus. 
From this tutorial you will learn how to:

* Implement new Problem class
* Implement new Model class
* Define and structure adequate configuration file 
* Train the mnewly defined model on the problem using training and validation sets
* Test the resulting model on the test set 



The MNIST Problem
-------------

MNIST is a classic dataset containing handwritten digits (0-9) with labels.
As such, the goal is to classify what is the content of the image, thus *Image Classification*.

.. figure:: ../img/mnist_examples.png
    :figwidth: 100 %
    :align: center
    
    Examples of MNIST images with corresponding labels (targets).


First, let us implement the MNIST problem class.
As it is an 'Image Classification' problem, we should derive our class from MI-Prometheus’s 
base ImageToClassProblem class. 


MNIST being available in TorchVision, we can write a simple wrapper and in __init__ create an instance of the dataset.MNIST class. 
We parse the values of the parameters from the configuration (batch size, data folder path, etc.) and set the loss function.
We also create the data_definitions dictionary indicating the type & shape of the content of one batch produced by our problem class. 


.. note::
    Please note that the '' MNIST'' problem downloads the required files on its own. 
    In this example we are relying on TorchVision MNIST dataset class, which checks if files exist and downloads them (if required) on the fly.
    That approach is as well the default behaviour of all Problems present in MI-Prometheus.

.. seealso:: 
    There are many datasets that are simply to big and should be downloaded a'priori.
    MI-Prometheus facilitates that with `ProblemInitializer`, a helper application that loads the problem, instantiates it and uses problem's definition 
    to download all the files into the path indicated by the configuration file.
    Most problem have ''~/data/problem_name'' path by default, but it can be changed by setting ''data_dir'' variable in problem section of the configuration file.

Then, we have to implement the __getitem__ and collate_fn, which are calling the adequate dataset.MNIST methods for returning samples. 
Please notice the keys in data_definition matching the keys in the __getitem__ function.


.. Wraps call to \texttt{\_\_getitem\_\_} from TorchVision, returns data consistent with \texttt{data\_definitions},
.. Collates all elements using \texttt{data\_definitions},
.. \texttt{\_\_getitem\_\_} \& \texttt{collate\_fn} enable utilization of PyTorch DataLoader for multiprocessing.

Image Classification problem use Cross Entropy for loss,
Define data_definitions, which describes the produced samples.



LeNet-5 Model
-------------

Now, let us focus on the LeNet-5 model, 

lenet5_architecture.png



Source: Y.LeCun, L.Bottou, Y.Bengio, and P.Haffner. `Gradient-based learning applied to document recognition
<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_, śProc. IEEE 86(11): 2278–2324, 1998. 


We will derive that model directly from the base Model class.
Analogically, in __init__, we define the data_definitions – here, we specify the image size the model expects and the shape of the predictions (logits) that it will return. 
Next, we create all the model’s trainable layers. The second important method is forward, which computes the logits on the basis of the inputs, by simply passing outputs from one layer to the other.




Configuration file
-------------------


Following, we need to create the corresponding configuration file (named mnist_lenet5.yaml) to execute the Trainer (Fig. 4). 
First, we define the Model section, which is straight forward, as the Model does not have any additional parameters. 
In the training section, we define the problem (name and required parameters), optimizer and terminal conditions. 
We select the Adam optimizer [KB14] and set an upper limit of 20 epochs. 
The section used by the Tester requires only the Problem’s definition and indicates that we use the test set.


Training
--------

In order to run the training, we must execute:

:command:`make`

    foo@bar$ mip-offline-trainer --c configs/vision/lenet5_mnist.yaml

This will result in:

    Info: Parsing the configs/vision/lenet5_mnist.yaml configuration file
    Loaded configuration from file configs/vision/lenet5_mnist.yaml
    Loaded (initial) configuration:
    ...

After the configuration is loaded, trainer starts creating objects and initializing variables according to the provided instructions.
First, as random seeds are not hardcoded in configuration file, the trainer sets their values to random::

    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Setting numpy random seed in training to: 2576900878
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Setting torch random seed in training to: 1158531658
    [2018-11-16 16:01:27] - WARNING - OfflineTrainer >>> GPU flag is disabled, using CPU.

Next, it instantiates problem used during training::

    [2018-11-16 16:01:27] - INFO - ProblemFactory >>> Loading the MNIST problem from miprometheus.problems.image_to_class.mnist
    [2018-11-16 16:01:27] - WARNING - MNIST >>> Upscaling the images to [32, 32]. Slows down batch generation.
    [2018-11-16 16:01:27] - INFO - SamplerFactory >>> Loading the SubsetRandomSampler sampler from torch.utils.data.sampler
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Problem for 'training' loaded (size: 60000)
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Sampler for 'training' created (size: 55000)

As you can see, the sampler that we indicated in the configuration file picked 55000 samples from the whole training set' leaving the remaining 5000 for validation.

Similarly, trainer instantiates dataloader, problem and sampler that will be used during the validation::

    [2018-11-16 16:01:27] - INFO - ProblemFactory >>> Loading the MNIST problem from miprometheus.problems.image_to_class.mnist
    [2018-11-16 16:01:27] - WARNING - MNIST >>> Upscaling the images to [32, 32]. Slows down batch generation.
    [2018-11-16 16:01:27] - INFO - SamplerFactory >>> Loading the SubsetRandomSampler sampler from torch.utils.data.sampler
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Problem for 'validation' loaded (size: 60000)
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Sampler for 'validation' created (size: 5000)

Next it creates the model::

    [2018-11-16 16:01:27] - INFO - ModelFactory >>> Loading the LeNet5 model from miprometheus.models.vision.lenet5
    [2018-11-16 16:01:27] - WARNING - Model >>> No parameter value was parsed from problem_default_values_
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>>
    ================================================================================
    Model name (Type)
    + Submodule name (Type)
        Matrices: [(name, dims), ...]
        Trainable Params: #
        Non-trainable Params: #
    ================================================================================
    LeNet5 (LeNet5)
    + conv1 (Conv2d)
    |   Matrices: [('weight', (6, 1, 5, 5)), ('bias', (6,))]
    |   Trainable Params: 156
    |   Non-trainable Params: 0
    |
    + maxpool1 (MaxPool2d)
    |   Matrices: []
    |   Trainable Params: 0
    |   Non-trainable Params: 0
    |
    + conv2 (Conv2d)
    |   Matrices: [('weight', (16, 6, 5, 5)), ('bias', (16,))]
    |   Trainable Params: 2416
    |   Non-trainable Params: 0
    |
    + maxpool2 (MaxPool2d)
    |   Matrices: []
    |   Trainable Params: 0
    |   Non-trainable Params: 0
    |
    + conv3 (Conv2d)
    |   Matrices: [('weight', (120, 16, 5, 5)), ('bias', (120,))]
    |   Trainable Params: 48120
    |   Non-trainable Params: 0
    |
    + linear1 (Linear)
    |   Matrices: [('weight', (84, 120)), ('bias', (84,))]
    |   Trainable Params: 10164
    |   Non-trainable Params: 0
    |
    + linear2 (Linear)
    |   Matrices: [('weight', (10, 84)), ('bias', (10,))]
    |   Trainable Params: 850
    |   Non-trainable Params: 0
    |

    Total Trainable Params: 61706
    Total Non-trainable Params: 0
    ================================================================================

Next, trainer loads/sets the terminal conditions::

    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Terminal conditions:
    ================================================================================
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Setting Loss Stop threshold to 0.01
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Partial Validation deactivated
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Setting the Epoch Limit to: 10
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Epoch size in terms of training episodes: 860
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Setting the Episode Limit to: 10000

This ends the setup phase.
When completed, trainer can finally start the training::

    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> Starting next epoch: 0
    [2018-11-16 16:01:27] - INFO - OfflineTrainer >>> loss 2.3083853722; episode 000000; epoch 00; acc 0.1250000000; batch_size 000064
    [2018-11-16 16:01:30] - INFO - OfflineTrainer >>> loss 0.2549477816; episode 000100; epoch 00; acc 0.9218750000; batch_size 000064
    [2018-11-16 16:01:33] - INFO - OfflineTrainer >>> loss 0.1247293055; episode 000200; epoch 00; acc 0.9687500000; batch_size 000064
    [2018-11-16 16:01:36] - INFO - OfflineTrainer >>> loss 0.1090276390; episode 000300; epoch 00; acc 0.9687500000; batch_size 000064
    [2018-11-16 16:01:39] - INFO - OfflineTrainer >>> loss 0.1786187887; episode 000400; epoch 00; acc 0.9375000000; batch_size 000064
    [2018-11-16 16:01:42] - INFO - OfflineTrainer >>> loss 0.1198359281; episode 000500; epoch 00; acc 0.9531250000; batch_size 000064
    [2018-11-16 16:01:45] - INFO - OfflineTrainer >>> loss 0.0421093963; episode 000600; epoch 00; acc 0.9843750000; batch_size 000064
    [2018-11-16 16:01:48] - INFO - OfflineTrainer >>> loss 0.0180228334; episode 000700; epoch 00; acc 1.0000000000; batch_size 000064
    [2018-11-16 16:01:51] - INFO - OfflineTrainer >>> loss 0.1346450299; episode 000800; epoch 00; acc 0.9531250000; batch_size 000064
    [2018-11-16 16:01:53] - INFO - OfflineTrainer >>> episode 000859; episodes_aggregated 000860; loss 0.1872173548; loss_min 0.0019642885; loss_max 2.3083853722; loss_std 0.2764583528; epoch 00; acc 0.9420421720; acc_min 0.1093750000; acc_max 1.0000000000; acc_std 0.0996650383; samples_aggregated 055000 [Epoch 0]
    [2018-11-16 16:01:53] - INFO - OfflineTrainer >>> Validating over the entire validation set (5000 samples in 79 episodes)
    [2018-11-16 16:01:55] - INFO - OfflineTrainer >>> episode 000859; episodes_aggregated 000079; loss 0.0667600185; loss_min 0.0000539126; loss_max 0.3059828281; loss_std 0.0653798506; epoch 00; acc 0.9810126424; acc_min 0.9375000000; acc_max 1.0000000000; acc_std 0.0169085916; samples_aggregated 005000 [Full Validation]
    [2018-11-16 16:01:55] - INFO - Model >>> Model and statistics exported to checkpoint ./experiments/MNIST/LeNet5/20181116_160127/models/model_best.pt

Fast-forwarding to the last epoch::

    [2018-11-16 16:06:02] - INFO - OfflineTrainer >>> Starting next epoch: 9
    [2018-11-16 16:06:05] - INFO - OfflineTrainer >>> loss 0.0502859205; episode 007800; epoch 09; acc 0.9687500000; batch_size 000064
    [2018-11-16 16:06:09] - INFO - OfflineTrainer >>> loss 0.0487646200; episode 007900; epoch 09; acc 0.9843750000; batch_size 000064
    [2018-11-16 16:06:12] - INFO - OfflineTrainer >>> loss 0.0395447724; episode 008000; epoch 09; acc 0.9843750000; batch_size 000064
    [2018-11-16 16:06:16] - INFO - OfflineTrainer >>> loss 0.0363486856; episode 008100; epoch 09; acc 0.9843750000; batch_size 000064
    [2018-11-16 16:06:19] - INFO - OfflineTrainer >>> loss 0.0027141620; episode 008200; epoch 09; acc 1.0000000000; batch_size 000064
    [2018-11-16 16:06:24] - INFO - OfflineTrainer >>> loss 0.0239426140; episode 008300; epoch 09; acc 1.0000000000; batch_size 000064
    [2018-11-16 16:06:27] - INFO - OfflineTrainer >>> loss 0.0041407160; episode 008400; epoch 09; acc 1.0000000000; batch_size 000064
    [2018-11-16 16:06:31] - INFO - OfflineTrainer >>> loss 0.1132633463; episode 008500; epoch 09; acc 0.9687500000; batch_size 000064
    [2018-11-16 16:06:35] - INFO - OfflineTrainer >>> episode 008599; episodes_aggregated 000860; loss 0.0667473748; loss_min 0.0000249359; loss_max 0.9916747212; loss_std 0.0870653242; epoch 09; acc 0.9845990539; acc_min 0.9375000000; acc_max 1.0000000000; acc_std 0.0157482121; samples_aggregated 055000 [Epoch 9]
    [2018-11-16 16:06:35] - INFO - OfflineTrainer >>> Validating over the entire validation set (5000 samples in 79 episodes)
    [2018-11-16 16:06:36] - INFO - OfflineTrainer >>> episode 008599; episodes_aggregated 000079; loss 0.0919304416; loss_min 0.0002728553; loss_max 0.4637385011; loss_std 0.1039550751; epoch 09; acc 0.9820016026; acc_min 0.9375000000; acc_max 1.0000000000; acc_std 0.0156427398; samples_aggregated 005000 [Full Validation]
    [2018-11-16 16:06:36] - INFO - OfflineTrainer >>>
    ================================================================================
    [2018-11-16 16:06:36] - INFO - OfflineTrainer >>> Training finished because Not converged: Epoch Limit reached
    [2018-11-16 16:06:36] - INFO - OfflineTrainer >>> Validating over the entire validation set (5000 samples in 79 episodes)
    [2018-11-16 16:06:37] - INFO - OfflineTrainer >>> episode 008599; episodes_aggregated 000079; loss 0.0919263810; loss_min 0.0000011193; loss_max 0.5043386221; loss_std 0.1023611873; epoch 09; acc 0.9820016026; acc_min 0.9375000000; acc_max 1.0000000000; acc_std 0.0146080535; samples_aggregated 005000 [Full Validation]
    [2018-11-16 16:06:37] - INFO - Model >>> Updated training status in checkpoint ./experiments/MNIST/LeNet5/20181116_160127/models/model_best.pt
    [2018-11-16 16:06:37] - INFO - OfflineTrainer >>> Experiment finished!


Test
----

Finally, we can use Tester to calculate the accuracy of the model on the test set. In order to do so we must indicate the checkpoint containing the model that we want to test::


    tomaszs-mbp:mi-prometheus tomaszkornuta$ mip-tester --m ./experiments/MNIST/LeNet5/20181116_160127/models/model_best.pt
    Info: Parsing the /Users/tomaszkornuta/Documents/GitHub/mi-prometheus/experiments/MNIST/LeNet5/20181116_160127/training_configuration.yaml configuration file
    Loaded configuration from file /Users/tomaszkornuta/Documents/GitHub/mi-prometheus/experiments/MNIST/LeNet5/20181116_160127/training_configuration.yaml
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting numpy random seed in testing to: 3971669138
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting torch random seed in testing to: 2013212003
    [2018-11-16 16:49:30] - WARNING - Tester >>> GPU flag is disabled, using CPU.

Analogically to trainer, Tester will next create dataloader, problem and sampler:

    [2018-11-16 16:49:30] - INFO - ProblemFactory >>> Loading the MNIST problem from miprometheus.problems.image_to_class.mnist
    [2018-11-16 16:49:30] - WARNING - MNIST >>> Upscaling the images to [32, 32]. Slows down batch generation.
    [2018-11-16 16:49:30] - INFO - SamplerFactory >>> The sampler configuration section is not present.
    [2018-11-16 16:49:30] - INFO - Tester >>> Problem for 'testing' loaded (size: 10000)
    [2018-11-16 16:49:30] - INFO - Tester >>> Setting the max number of episodes to: 157

And model. Please notice that this time it has also loaded model parameters from checkpoint 

    [2018-11-16 16:49:30] - INFO - ModelFactory >>> Loading the LeNet5 model from miprometheus.models.vision.lenet5
    [2018-11-16 16:49:30] - WARNING - Model >>> No parameter value was parsed from problem_default_values_
    [2018-11-16 16:49:30] - INFO - Model >>> Imported LeNet5 parameters from checkpoint from 2018-11-16 16:04:28.446649 (episode: 5159, loss: 0.05859316140413284, status: Not converged: Epoch Limit reached)
    [2018-11-16 16:49:30] - INFO - Tester >>>


..note::
    Trainer has the same capability.
    If you want to load and e.g. finetune your model use option --m(odel).

After the setup phase, tester starts its experiment::

    [2018-11-16 16:49:30] - INFO - Tester >>> Testing over the entire test set (10000 samples in 157 episodes)
    [2018-11-16 16:49:30] - INFO - Tester >>> loss 0.0415740199; episode 000000; acc 0.9843750000; batch_size 000064 [Partial Test]
    [2018-11-16 16:49:31] - INFO - Tester >>> loss 0.0111131836; episode 000100; acc 1.0000000000; batch_size 000064 [Partial Test]
    [2018-11-16 16:49:32] - INFO - Tester >>>
    ================================================================================
    [2018-11-16 16:49:32] - INFO - Tester >>> Test finished
    [2018-11-16 16:49:32] - INFO - Tester >>> episode 000157; episodes_aggregated 000157; loss 0.0711115003; loss_min 0.0005262249; loss_max 0.6216378212; loss_std 0.0811631531; acc 0.9837778807; acc_min 0.9218750000; acc_max 1.0000000000; acc_std 0.0148953591; samples_aggregated 010000 [Full Test]