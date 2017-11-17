#!/bin/bash
# Set up a Google Compute Engine instance with GPU 

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install zip
sudo apt-get install gcc
sudo apt-get install build-essential
sudo apt-get install linux-headers-$(uname -r)

mkdir setup
cd setup

# Copy cuda device driver files from bucket
gsutil cp gs://neon-opus-178512-data/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb .
gsutil cp gs://neon-opus-178512-data/cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb .

sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb

sudo apt-get update
sudo apt-get install cuda

# Add lines to the end of .profile or .bashrc
echo 'export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Copy cudnn device driver files from bucket
gsutil cp gs://neon-opus-178512-data/libcudnn6_6.0.21-1+cuda8.0_amd64.deb .
gsutil cp gs://neon-opus-178512-data/libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb .
gsutil cp gs://neon-opus-178512-data/libcudnn6-doc_6.0.21-1+cuda8.0_amd64.deb .

sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6-doc_6.0.21-1+cuda8.0_amd64.deb

sudo apt-get update
sudo apt-get install libcupti-dev

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

cd ~

conda create -n ml python=3.6
source activate ml
conda install jupyter pandas numpy scipy scikit-image
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl
pip install keras h5py

# Install ffmpeg
sudo apt-get install ffmpeg
