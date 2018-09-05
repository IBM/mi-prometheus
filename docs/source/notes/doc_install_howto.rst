How to install pytorch and prometheus from scratch ?
================================================== 
@author: Ryan L. McAvoy
 
Guidelines & examples 
-------------------------------------------


Installing on an empty server (Ubuntu 16.04). 

The following are examples and should be modified to suit your preferences.

In the home directory, make an executable bash script containing following and then run it.

    #!/bin/bash -x
    
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
    
    /opt/conda/bin/python3 -m pip install --user virtualenv
    
    /opt/conda/bin/python3 -m virtualenv env
    
    echo "export PATH=/opt/conda/bin:$PATH" >> env/bin/activate
    
    source env/bin/activate
    
    /opt/conda/bin/conda install numpy pyyaml setuptools mkl mkl-include cmake cffi typing
    /opt/conda/bin/conda clean -ya
    
    rm -rf pytorch
    export  && \
        git clone --recursive https://github.com/pytorch/pytorch && \
        cd pytorch && \
        git checkout v0.4.0 && \
        git submodule init && \
        git submodule update && \
        CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
        CFLAGS="-march=native" CXXFLAGS="-O3 -march=native" /opt/conda/bin/python3 setup.py install
    cd ..
    rm -rf pytorch

In the home directory, git clone mi-prometheus from the github repository. Run the following script. 
If you did not run the previous script then you will need to modify the echo command so that it appends to either a different activate file or the .bashrc

    #!/bin/bash -x
    
    echo "export PYTHONPATH='${PYTHONPATH}:~/mi-prometheus/'" >> ~/env/bin/activate
    
    conda install -c conda-forge torchvision
    pip install torchtext
    conda install -c conda-forge tensorboardX
    conda install pyyaml matplotlib ffmpeg
    conda install sphinx sphinx_rtd_theme
    conda install tqdm
    conda install progressbar2
    
    #seems to come by default but doesn't hurt to be sure
    conda install nltk
    conda install h5py
    conda install pandas
    conda install pillow
    conda install six
