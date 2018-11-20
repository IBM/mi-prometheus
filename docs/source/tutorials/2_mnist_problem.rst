The MNIST Problem
------------------

MNIST is a classic dataset containing handwritten digits (0-9) with labels.
As such, the goal is to classify what is the content of the image, thus *Image Classification*.
The images are grayscale 28x28 pixels, whereas the whole dataset consits of two subsets:

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


Next let us implement the ``__init__`` method::

    def __init__(self, params_):
        # Call base class constructors.
        super(MNIST, self).__init__(params_, 'MNIST')

Let us start with default parameters indicating the folder containing files with MNIST images and labels; and whether we will use training or test set::


        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/mnist',
                                        'use_train_data': True
                                        })

.. note::
    Add parameters of the registry added in the code using `add_default_params`  will be overwritten by the values read from configuration file.


We parse the values of the parameters from the configuration (batch size, data folder path, etc.) and set the loss function::

        # Get absolute path.
        data_folder = os.path.expanduser(self.params['data_folder'])

        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']

To make the problem more flexible, we will add an optional transformation.
If `resize` will be set in configuration file, then problem will rescale every image to the provided [HEIGHT WIDTH]::

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

Otherwise we will return tensors with images holding the orginal sizes::

        else:
            # Default MNIST settings.
            self.width = 28
            self.height = 28
            self.num_channels = 1
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])



Next we should define default values (e.g. the size of the image), that will be passed to the model::

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': 10,
                               'num_channels': self.num_channels,
                               'width': self.width,
                               'height': self.height}

We also create the `data_definitions` dictionary indicating the type & shape of the content of one batch produced by our problem class.::

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

.. note:: 
    Please note that -1 is a special value indicating unknown, potentially varying dimension. In MI-Prometheus we are following batch major standard.
    For example, in the case if images the dimensions are [BATCH SIZE x NUMBER OF CHANNELS X HEIGHT x WIDTH], whereas for sequence of words it will be [BATCH SIZE x SEQUENCE LENGTH x EMBEDDING SIZE].

.. _dataset.MNIST: https://pytorch.org/docs/stable/torchvision/datasets.html#mnist


As MNIST is being available in TorchVision, we will reuse it and wrap it in our class.
For that purpose we will create an instance of the dataset.MNIST_  class::

        # load the dataset
        self.dataset = datasets.MNIST(root=data_folder, train=self.use_train_data, download=True,
                                      transform=transform)

.. note::
    Please note that the dataset.MNIST_ object downloads the required files on its own. 
    Such an approach, when a problem checks if required files exist (and downloads them on the fly if absent), was adapted as the default behaviour of all Problems present in MI-Prometheus.

.. seealso:: 
    There are many datasets that are simply to big and should be downloaded a'priori.
    MI-Prometheus facilitates that with `ProblemInitializer`, a helper application that loads the problem, instantiates it and uses problem's definition 
    to download all the files into the path indicated by the configuration file.
    Most problems have ''~/data/problem_name'' path set by default, but it can be changed by setting ''data_dir'' variable in problem section of the configuration file.


We also need to set the length of the dataset and labels, what ends the initialization of the problem::

        # Set length.
        self.length = len(self.dataset)

        # Class names.
        self.labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')

.. note::
    :py:class:`~miprometheus.problems.ImageTextToClassProblem` class sets `self.loss_function` to cross entropy loss. We will use it as default here.

Next, we have to implement the ''__getitem__'' method, which once agains will simply wrap the adequate dataset.MNIST_ method.
Please notice the keys in data_definition matching the keys in the __getitem__ function::

   def __getitem__(self, index):
        # Get image and target.
        img, target = self.dataset.__getitem__(index)
  
        # Digit label.
        label = self.labels[target.data]

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict['images'] = img
        data_dict['targets'] = target
        data_dict['targets_label'] = label
        return data_dict

.. note::
    A single image stored under the data_dict['images'] key is a 3D tensor [NUMBER OF CHANNELS X HEIGHT x WIDTH].


Finally, we need to implement the ''collate_fn'', which is responsible for putting together all the samples from a given batch (list) into a single tensor,
in this case of size [BATCH SIZE x NUMBER OF CHANNELS X HEIGHT x WIDTH]::


    def collate_fn(self, batch):
        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(MNIST, self).collate_fn(batch).values())})

Please notte that both ''__get_item__'' and ''collate_fn'' rely on ''data_definitions that we created during problem initialization.

