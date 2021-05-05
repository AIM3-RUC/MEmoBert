FROM nvidia/cuda:10.1-devel-ubuntu18.04

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
PYTORCH_VERSION=1.6.0
TORCHVISION_VERSION=0.7.0
CUDNN_VERSION=7.6.5.32-1+cuda10.1
NCCL_VERSION=2.7.8-1+cuda10.1

# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
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
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
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


# Install Open MPI
mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 \
         pip install --no-cache-dir horovod[all-frameworks] && \
    ldconfig
apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

最后需要安装apex.
git clone xxxx
+ Bug4.1 https://github.com/NVIDIA/apex/issues/802 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

然后根据提示安装一些必要的包

---- 终于配置好了----