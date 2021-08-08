FROM nvidia/cuda:10.1-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
PYTORCH_VERSION=1.6.0
TORCHVISION_VERSION=0.7.0
CUDNN_VERSION=7.6.5.32-1+cuda10.1
NCCL_VERSION=2.7.8-1+cuda10.1

# Python 3.7 is supported by Ubuntu Bionic out of the box
python=3.7
PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates 
        
libcudnn7=${CUDNN_VERSION} 
/data2/zjm/tools/cudnn-10.1-linux-x64-v7.6.5.32.tgz

apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev 

apt-get install python3-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers

ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install TensorFlow, Keras, PyTorch and MXNet
pip install future typing packaging
pip install h5py

PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu101/torch-${PYTORCH_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu101/torchvision-${TORCHVISION_VERSION}%2Bcu101-${PYTAGS}-linux_x86_64.whl


# Install Open MPI --Update to 4.1
cd /data7/emobert/resources/openmpi 
wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.0.tar.gz
tar zxf openmpi-4.1.0.tar.gz
cd openmpi-4.1.0
./configure --prefix=/usr/local
make -j $(nproc) all
make -j $(nproc) install
ldconfig

OPENMPI_VERSION=4.1.0

# Install Horovod, temporarily using CUDA stubs
ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
   HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir horovod==0.22.1 &&\
    ldconfig

[Bug1] Horovod build with GPU support was requested but this MXNet installation does not support CUDA.
HOROVOD_WITHOUT_MXNET=1

# 这个也需要装的
apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd


最后需要安装apex.
git clone https://github.com/NVIDIA/apex.git 
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
cd apex && \
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

然后根据提示安装一些必要的包

---- 终于配置好了----