How to keep this documentation up to date ?
==================================================
@author: Vincent Marois

**It is of high priority that the documentation of MI Prometheus is kept up-to-date as the code base evolves.
Good code without good documentation is not useful!**

Guidelines & examples
-------------------------------------------

Here is a quick how-to guide on how to keep this documentation up-to-date.

- The documentation source files are contained in `mi-prometheus/docs/source`. This directory contains:

    - `conf.py`: Configuration file for the Sphinx documentation builder.
    - `index.rst`: master table of content document for the entire documentation.
    - `models.rst`: master table of content document for the `models/` directory.
    - `problems.rst`: master table of content document for the `problems/` directory.
    - `misc.rst`: master table of content document for the `misc/` directory. [1]_
    - `notes/`: contains global information about the documentation pages (e.g. this page and the License).
    - `workers/`: contains the workers documentation pages.

You should not have to edit the `conf.py` file when doing changes to the code base. We mainly have to maintain the `.rst` files and the **import** lines in the `__init__.py` files.

The `.rst` files are written using the reStructuredText plaintext markup syntax. This is where you can define the hierarchy of the table of content. Here is an example of hierarchy:

::

    Models  # This is a title
    =============================

    .. automodule:: models
    .. currentmodule:: models

    Model  # this is a subtitle
    ---------------------------------

    .. autoclass:: Model
        :members:

    :hidden:`CNN_LSTM_VQA`  # this is a subsubtitle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: models.cnn_lstm_vqa
        :members:

    SequentialModel # this is a subtitle
    ----------------------------------------
    ..  currentmodule:: models
    .. autoclass:: SequentialModel
        :members:

    :hidden:`DWM` # this is a subsubtitle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: models.dwm
        :members:


Do not hesitate to frequently to refer to the reStructuredText guide_ for more information on the formatting.

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

- Finally, we have to rebuild the `.html` pages from the `.rst` files. This is done by executing the script `docgen.sh` in `mi-prometheus/`:

  >>> ./docgen.sh

  Make sure the packages `sphinx` and `sphinx_rtd_theme` are installed in your `Python` environment.
  To correctly create the documentation pages, `sphinx` will also require that packages like torch,
  torchvision, torchtext, matplotlib (pyyaml, pillow, h5py, progressbar2, nltk...) are also present in the environment.
  The reason is that `sphinx` actually imports the `mi-prometheus` packages to pull the docstrings. So we need to make sure
  that all packages on top of which `mi-prometheus` is built are present in the same environment.



Some quotes about Code Documentation
-------------------------------------------
::

    "Always code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live." - John F. Woods
    "Ink is better than the best memory." - Chinese proverb
    "The documentation needs documentation." - a Bellevue Linux Users Group member, 2005





.. [1] #TODO: Rename the `misc/` directory to `utilities/`.