.. _installation:

Installation
===================
`@author: Vincent Marois`


To use Mi-Prometheus, we need to first install PyTorch. Refer to the official installation guide_ of PyTorch for its installation.
It is easily installable via conda_, or you can compile it from source to optimize it for your machine.

.. _guide: https://github.com/pytorch/pytorch#installation

Mi-Prometheus is not yet available as a pip_ package, or on conda_. We are currently working on it, but would like to ensure a
certain stability first. The `setup.py` script is available and should be used for installation of Mi-Prometheus.

.. _conda: https://pypi.org/
.. _pip: https://pypi.org/

So you can clone the repository, checkout out a particular branch if wanted and run `python setup.py install`.
This command will install all dependencies of Mi-Prometheus via pip_.

**Please note that it will not install PyTorch**, as we have observed inconsistent and erratic errors when we were installing it from pip_.
Use conda_, or compile from source instead. The indicated way by the PyTorch team is conda_.

The `setup.py` will register Mi-Prometheus as a package in your `Python` environment so that you will be able to `import` it:

  >>> import miprometheus as mip

And then you will be able to access the API as regular:

  >>> datadict = mip.utils.DataDict()

etc.

The `setup.py` also creates aliases for the workers, so that you can use them as regular commands:

  >>> mip-offline-trainer --c path/to/your/config/file

The available commands are:

    - mip-offline-trainer
    - mip-online-trainer
    - mip-tester
    - mip-grid-trainer-cpu
    - mip-grid-trainer-gpu
    - mip-grid-tester-cpu
    - mip-grid-tester-gpu
    - mip-grid-analyzer

Each command executes the worker of the same name. Use `--h` to see the available flags for each command.

You can then delete the cloned repository and use these commands to run a particular worker with your configurations files.

**Please note** that we provide multiple configuration files in `configs/` (which we use daily for our research & development).
Feel free to copy this folder somewhere and use these files as you would like. We are investigating on how we could make these files usable in an easier way.

