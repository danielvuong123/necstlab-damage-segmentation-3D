#!/bin/bash

echo "Running apt-get update"
sudo apt-get update

echo "Running install build-essentials"
sudo apt-get install -y build-essential

# install cuda for tf2.1
# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
# sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-wsl-ubuntu-11-2-local_11.2.2-1_amd64.deb
# sudo dpkg -i cuda-repo-wsl-ubuntu-11-2-local_11.2.2-1_amd64.deb
# sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-2-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda
echo "Executing export of cuda to path"
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}



# install cudnn
# wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz
# tar -xzvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
# sudo cp cuda/include/cudnn.h /usr/local/cuda/include
# sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

#Setup CUPTI path
echo "Exporting cuda to library path"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.2/lib64

echo "Exporting local system path to remote path"
export PATH="/home/dcvuong/.local/bin:$PATH"
# ls /..
# ls
# ls /..

# install needed packages
echo "installing python packages"
sudo apt-get install -y cmake \
    git \
    python3-setuptools \
    python3-dev \
    python3-pip \
    libopencv-dev \
    htop \
    tmux \
    tree \
    p7zip-full

echo "installing remaining packages."
pip3 install -U pip
pip3 install --upgrade setuptools
pip3 uninstall crcmod -y
pip3 install --no-cache-dir crcmod
pip3 install --upgrade pyasn1
#The next line only works if your current username (In your command line) and the gcp username match

echo "Executing last step, installing requirements.txt"
cd necstlab-damage-segmentation && python3 -m pip install -r requirements.txt
