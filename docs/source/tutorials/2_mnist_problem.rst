The MNIST Problem
------------------

MNIST is a classic dataset containing handwritten digits (0-9) with labels.
As such, the goal is to classify what is the content of the image, thus *Image Classification*.
The images are 28x28 pixel, and the whole dataset consits of two subsets:

    * Training set (60.000 images with corresponding labels)
    * Test set (10.000 images with labels)


.. figure:: ../img/mnist_examples.png
    :figwidth: 100 %
    :align: center
    
    Examples of MNIST images with corresponding labels (targets).


So let us implement the MNIST problem class.
As it is an **Image Classification** problem, we should derive our class from MI-Prometheus’s 
base :py:class:`~miprometheus.problems.ImageToClassProblem` class::

    import os
    import torch
    from torchvision import datasets, transforms

    from miprometheus.utils.data_dict import DataDict
    from miprometheus.problems.image_to_class.image_to_class_problem import ImageToClassProblem


    class MNIST(ImageToClassProblem):


.. note::
    MI-Prometheus' problem class hierarchy and naming conventions (in the case of intermediate classes) follows **Input** **To** **Output** pattern.
    In case of leaves, we keep it simple and use names that correspond to problem or dataset name established in the literature.
    For example, :py:class:`~miprometheus.problems.CLEVR` inherits from
    :py:class:`~miprometheus.problems.ImageTextToClassProblem` class. 


As MNIST is being available in TorchVision, we will reuse it and write a simple wrapper class.
For that purpose in ``__init__`` create an instance of the dataset.MNIST class::

    def __init__(self, params_):
        # Call base class constructors.
        super(MNIST, self).__init__(params_, 'MNIST')

        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/mnist',
                                        'use_train_data': True
                                        })

        # Get absolute path.
        data_folder = os.path.expanduser(self.params['data_folder'])

        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']

        # Add transformations depending on the resizing option.
        if 'resize' in self.params:
            # Check the desired size.
            if len(self.params['resize']) != 2:
                self.logger.error("'resize' field must contain 2 values: the desired height and width")
                exit(-1)

            # Output image dimensions.
            self.height = self.params['resize'][0]
            self.width = self.params['resize'][1]
            self.num_channels = 1

            # Up-scale and transform to tensors.
            transform = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

            self.logger.warning('Upscaling the images to [{}, {}]. Slows down batch generation.'.format(
                self.width, self.height))

        else:
            # Default MNIST settings.
            self.width = 28
            self.height = 28
            self.num_channels = 1
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': 10,
                               'num_channels': self.num_channels,
                               'width': self.width,
                               'height': self.height}

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

        # load the dataset
        self.dataset = datasets.MNIST(root=data_folder, train=self.use_train_data, download=True,
                                      transform=transform)

        # Set length.
        self.length = len(self.dataset)

        # Class names.
        self.labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')



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

