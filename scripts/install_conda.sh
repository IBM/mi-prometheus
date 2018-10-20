#!/usr/bin/env bash
# This script installs Anaconda 5.3.
# Tested on a docker image of ubuntu:16.04 on 10/11/18.

apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates && \
    apt-get clean

wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh

chmod +x ~/anaconda.sh

rm -rf /opt/conda
~/anaconda.sh -b -p /opt/conda
rm ~/anaconda.sh

echo "# added by Anaconda3 installer" >> ~/.bashrc
echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc

# this cannot be done inside a script, hence the user has to do it manually.
echo "Installation successful. Please run the command below to add conda to the path."
echo "source ~/.bashrc"