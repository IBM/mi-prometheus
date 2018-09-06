#!/bin/bash -x

echo "export PYTHONPATH='${PYTHONPATH}:~/mi-prometheus/'" >> /opt/conda/bin/activate

pip install torchvision torchtext tensorboardX
conda install pyyaml matplotlib ffmpeg sphinx sphinx_rtd_theme tqdm progressbar2 nltk h5py pandas pillow six pyqt

#  Make sure the packages `sphinx` and `sphinx_rtd_theme` are installed in your `Python` environment.
#  To correctly create the documentation pages, `sphinx` will also require that packages like torch,
#  torchvision, torchtext, matplotlib (pyyaml, pillow, h5py, progressbar2, nltk...) are also present in the environment.
#  The reason is that `sphinx` actually imports the `mi-prometheus` packages to pull the docstrings. So we need to make sure
#  that all packages on top of which `mi-prometheus` is built are present in the same environment.
