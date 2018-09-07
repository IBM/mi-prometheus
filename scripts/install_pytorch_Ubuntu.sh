#!/bin/bash -x
# this scripts install Anaconda & Pytorch 0.4.0 from source.

apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        g++ \
        git \
        wget \
        ca-certificates && \
    apt-get clean


sudo apt install build-essential

wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh

chmod +x ~/anaconda.sh

rm -rf /opt/conda
~/anaconda.sh -b -p /opt/conda
rm ~/anaconda.sh

# create a conda env.
conda create -n mi-prometheus python=3.6

# Not using virtualenv as it breaks installs of matplotlib on Mac's according to this help page
# https://matplotlib.org/faq/osx_framework.html
#/opt/conda/bin/python3 -m pip install --user virtualenv
#/opt/conda/bin/python3 -m virtualenv env

echo "export PATH=/opt/conda/bin:$PATH" >> /opt/conda/bin/activate

source activate mi-prometheus
#source env/bin/activate

#/opt/conda/bin/conda install numpy pyyaml setuptools mkl mkl-include cmake cffi typing
#/opt/conda/bin/conda clean -ya

conda install numpy pyyaml setuptools mkl mkl-include cmake cffi typing
conda clean -ya

rm -rf pytorch
export  && \
    git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git checkout v0.4.0 && \
    git submodule init && \
    git submodule update && \ 
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    CFLAGS="-march=native" CXXFLAGS="-O3 -march=native" python3 setup.py install
cd ..
rm -rf pytorch
