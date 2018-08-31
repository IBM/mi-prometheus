How to keep this documentation up to date ?
==================================================
**It is of high priority that the documentation of MI Prometheus is kept up-to-date as the code base evolves.
Good code without good documentation is not useful!**

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

    Models
    ========

    .. automodule:: models
    .. currentmodule:: models

    Model
    ----------

    .. autoclass:: Model
        :members:

    :hidden:`CNN_LSTM_VQA`
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: models.cnn_lstm_vqa
        :members:

    SequentialModel
    ----------------------
    ..  currentmodule:: models
    .. autoclass:: SequentialModel
        :members:

    :hidden:`DWM`
    ~~~~~~~~~~~~~~~~
    .. automodule:: models.dwm
        :members:


Do not hesitate to frequently to refer to the reStructuredText guide_ for more information on the formatting.

.. _guide: http://docutils.sourceforge.net/docs/user/rst/quickref.html

When adding a new module (`.py` file), class or function in the code base, please do the following:

- First, update the *closest* `__init__.py` file in the `mi-prometheus` code hierarchy to **import** the new structures you have written.
  For instance, let's assume you have written a file `master_algorithm.py` that contains a class named `MeaningOfLife`, and that this file is located in `dir1/dir2/`.
  In the `__init__.py` file of `dir2/`, add the following lines:

  >>> from .master_algorithm import MeaningOfLife
  >>> __all__ = [..., 'MeaningOfLife']

  The first line imports the new class from the file you wrote. The second line adds it to the list of public objects of that module., as interpreted by import *. It overrides the default of hiding everything that begins with an underscore.

make sure `sphinx` and `sphinx_rtd_theme` are installed in your python env. Will also require packages like torch,
torchvision, torchtext, matplotlib, pyyaml, pillow, h5py, progressbar2, nltk

the source .rst files are in docs/source
conf.py ->
index.rst ->
models.rst >
problems.rst ->
misc.rst ->

workers/ -> contains documentation of the different workers
notes/ -> contains this file + the license

- if adding a new module or class or function -> add import lines in the closest __init__ file

- add the entry in the corresponding .rst file (pay attention to the hierarchy)
-> show example

make slide to present to team

Some quotes about Code Documentation
-------------------------------------------
::

    "Always code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live." - John F. Woods
    "Ink is better than the best memory." - Chinese proverb
    "The documentation needs documentation." - a Bellevue Linux Users Group member, 2005





.. [1] #TODO: Rename the `misc/` directory to `utilities/`.