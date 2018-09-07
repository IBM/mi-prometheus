#!/bin/bash -x

# Add mi-prometheus/ to PYTHONPATH so that Python can find the modules & packages
# This is done dynamically: when running `source activate mi-prometheus`, PYTHONPATH will be updated to
# include mi-prometheus/. When running `source deactivate mi-prometheus`, it will be set back to its previous value.

# to set it globally for all conda envs
#echo "export PYTHONPATH='${PYTHONPATH}:~/mi-prometheus/'" >> /opt/conda/bin/activate

# just to make sure that we have activated the conda env: shouldn't hurt if already activated
source activate mi-prometheus

cd $CONDA_PREFIX

# create a script that is called every time we activate the conda env
mkdir -p ./etc/conda/activate.d
touch ./etc/conda/activate.d/env_vars.sh

# create a script that is called every time we deactivate the conda env
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/deactivate.d/env_vars.sh

# save the previous value of PYTHONPATH
PREV_PYTHONPATH=$PYTHONPATH

# set PYTHONPATH to include mi-prometheus/ when activating the mi-prometheus env
echo '#!/bin/sh' >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH='${PYTHONPATH}:~/mi-prometheus/'" >> ./etc/conda/activate.d/env_vars.sh

# reset PYTHONPATH to its previous value when deactivating the mi-prometheus env
echo '#!/bin/sh' >> ./etc/conda/deactivate.d/env_vars.sh
echo "export PYTHONPATH='${PREV_PYTHONPATH}'" >> ./etc/conda/deactivate.d/env_vars.sh

# install torchvision from source
cd /tmp
git clone https://github.com/pytorch/vision.git
cd vision/
python setup.py install
cd ..
rm -rf vision/

# install the remaining packages
pip install torchtext tensorboardX
conda install pyyaml matplotlib ffmpeg sphinx sphinx_rtd_theme tqdm progressbar2 nltk h5py pandas pillow six pyqt

#  Make sure the packages `sphinx` and `sphinx_rtd_theme` are installed in your `Python` environment.
#  To correctly create the documentation pages, `sphinx` will also require that packages like torch,
#  torchvision, torchtext, matplotlib (pyyaml, pillow, h5py, progressbar2, nltk...) are also present in the environment.
#  The reason is that `sphinx` actually imports the `mi-prometheus` packages to pull the docstrings. So we need to make sure
#  that all packages on top of which `mi-prometheus` is built are present in the same environment.
