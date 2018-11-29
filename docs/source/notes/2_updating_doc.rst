Updating the documentation
==============================
`@author: Vincent Marois`

**It is of high priority that the documentation of MI Prometheus is kept up-to-date as the code base evolves.
Good code without good documentation is not useful!**

Guidelines & examples
-------------------------------------------

Here is a quick how-to guide on how to keep this documentation up-to-date.

- The documentation source files are contained in `mi-prometheus/docs/source`. This directory contains:

    - `conf.py`: Configuration file for the Sphinx documentation builder.
    - `index.rst`: master table of content document for the entire documentation.
    - `models.rst`: master table of content document for the `models` package.
    - `problems.rst`: master table of content document for the `problems` package.
    - `utils.rst`: master table of content document for the `utils` package.
    - `workers.rst`: master table of content document for the `workers` package.
    - `notes/`: contains global information about the documentation pages (e.g. this page).

Other folders (e.g. `tutorials/`) should be coming as the documentation grows.


You should not have to edit the `conf.py` file when doing changes to the code base. We mainly have to maintain the `.rst` files and the **import** lines in the `__init__.py` files.

The `.rst` files are written using the reStructuredText plaintext markup syntax. This is where you can define the hierarchy of the table of content. Here is an example of hierarchy:

::

    Models  # This is a title
    =============================

    .. automodule:: miprometheus.models


    Model  # this is a subtitle
    ---------------------------------

    .. autoclass:: Model
        :members:
        :special-members:
        :exclude-members: __dict__,__weakref__

    :hidden:`CNN + LSTM` # This is a subsubtitle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: miprometheus.models.vqa_baselines.cnn_lstm
        :members:
        :special-members:
        :exclude-members: __dict__,__weakref__

    SequentialModel # this is a subtitle
    ----------------------------------------
    .. autoclass:: SequentialModel
        :members:
        :special-members:
        :exclude-members: __dict__,__weakref__

    :hidden:`DWM` # this is a subsubtitle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: miprometheus.models.dwm
        :members:
        :special-members:
        :exclude-members: __dict__,__weakref__


Do not hesitate to frequently refer to the reStructuredText guide_ for more information on the formatting.

.. _guide: http://docutils.sourceforge.net/docs/user/rst/quickref.html

When adding a new module (`.py` file), class or function in the code base, please do the following:

- First, update the **closest** `__init__.py` file in the `mi-prometheus` code hierarchy to **import** the new structures you have written.
  For instance, let's assume you have written a file `master_algorithm.py` that contains a class named `MeaningOfLife`, and that this file is located in `dir1/dir2/`.
  In the `__init__.py` file of `dir2/`, add the following lines:

  >>> from .master_algorithm import MeaningOfLife
  >>> __all__ = [..., 'MeaningOfLife']

  The first line imports the new class from the file you wrote. The second line adds it to the list of public objects of that module, so that in the `__init__.py` file of `dir1/`, the line:

  >>> from .dir2 import *

  will properly import `MeaningOfLife`.

  So, if you are adding one `.py` file (called a module) to a directory that already contains a `__init__.py` file (this dir is then called a package), you only have to edit this `__init__`.

  If you are adding several directories and subdirectories, you have to update the several `__init__.py` files by traversing the hierarchy from innermost to outermost (simplest way to ensure you are not forgetting anything).

  **NOTE**: This may change in the future if our guidelines on the import lines change.

- Second, we have to update the corresponding `.rst` file to include this new module/class in the table of content.

  The main markers (called directives by the reStructuredText syntax) are

  ::

      .. automodule:: dir1.dir2.master_algorithm
          :members:

  and

  ::

      .. autoclass:: dir1.dir2.master_algorithm.MeaningOfLife
          :members:

  If you are adding only functions / methods within an existing class or module and if the reference to that class or module already exists in the table of content, you should not have to edit anything.
  Reconstructing the `.html` pages from the `.rst` files should automatically pull the corresponding docstrings.

  If you are adding a new class or module, then you have to add the reference in the table of content.
  The current (root) module in the `.rst` file should be indicated at the top of the file with the directive:

  ::

      .. currentmodule:: dir1

  So you just have to add the above sections at the location you want in the table of content hierarchy.

- Finally, we have to rebuild the `.html` pages from the `.rst` files. This is done by readthedocs_ when we do a commit to our repository.


**NOTE**: We are not using the `setup.py` to build the documentation, but rather using mocking_ to ignore the dependencies.
The reason is as follows:

    - The installation of the framework (through `python setup.py install`) can be resource intensive and the docker backend of readthedocs is constrained in terms of memory.
    - The documentation build should be pretty fast. Hence, avoiding dealing with dependencies is better.


Please refer to the `readthedocs.yml` file to see the configuration for the documentation build.

.. _readthedocs: https://readthedocs.org/projects/mi-prometheus/
.. _mocking: https://docs.python.org/3/library/unittest.mock.html

Some quotes about Code Documentation
-------------------------------------------
::

    "Always code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live." - John F. Woods
    "Ink is better than the best memory." - Chinese proverb
    "The documentation needs documentation." - a Bellevue Linux Users Group member, 2005
