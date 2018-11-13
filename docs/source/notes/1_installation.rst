Installation
===================
`@author: Vincent Marois & Tomasz Kornuta`


To use Mi-Prometheus, we need to first install PyTorch_. Refer to the official installation guide_ of PyTorch for its installation.
It is easily installable via conda_, or you can compile it from source to optimize it for your machine.

Mi-Prometheus is not (yet) available as a pip_ package, or on conda_.
However, we provide the `setup.py` script and recommend to use it for the installation of Mi-Prometheus.
First please clone the MI-Prometheus repository::

  git clone git@github.com:IBM/mi-prometheus.git
  cd mi-prometheus/

Then, install the dependencies by running::

  python setup.py install

This command will install all dependencies of Mi-Prometheus via pip_.
If you plan to develop and introduce changes, please call the following command instead::

  python setup.py develop

This will enable you to change the code of the existing problems/models/workers and still be able to run them by calling the associated 'mip-*' commands.
More in that subject can be found in the following blog post on dev_mode_.

.. _guide: https://github.com/pytorch/pytorch#installation
.. _PyTorch: https://github.com/pytorch/pytorch
.. _conda: https://anaconda.org/pytorch/pytorch
.. _pip: https://pip.pypa.io/en/stable/quickstart/
.. _dev_mode: https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode


**Please note that it will not install PyTorch**, as we have observed inconsistent and erratic errors when we were installing it from pip_.
Use conda_, or compile it from source instead. The indicated way by the PyTorch team is conda_.

The `setup.py` will register Mi-Prometheus as a package in your `Python` environment so that you will be able to `import` it:

  >>> import miprometheus as mip

And then you will be able to access the API, for instance:

  >>> datadict = mip.utils.DataDict()

Additionally, `setup.py` also creates aliases for the workers, so that you can use them as regular commands::

  mip-offline-trainer --c path/to/your/config/file

The currently available commands are:

    - mip-offline-trainer
    - mip-online-trainer
    - mip-tester
    - mip-grid-trainer-cpu
    - mip-grid-trainer-gpu
    - mip-grid-tester-cpu
    - mip-grid-tester-gpu
    - mip-grid-analyzer
    - mip-index-splitter

Use `--h` to see the available flags for each command.

You can then delete the cloned repository and use these commands to run a particular worker with your configurations files.

**Please note** that we provide multiple configuration files in `configs/` (which we use daily for our research & development).
Feel free to copy this folder somewhere and use these files as you would like. We are investigating on how we could make these files usable in an easier way.

