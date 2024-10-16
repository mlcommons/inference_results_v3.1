# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    cmake \
    wget \
    findutils \
    build-essential \
    gcc-12 \
    g++-12 \
    libzstd-dev \
    libgtk2.0-dev \
    libgl1 \
    libxml2-dev \
    zlib1g-dev \
    libdata-dumper-simple-perl \
    && rm -rf /var/lib/apt/lists/* &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 10 && \ 
    update-alternatives --set gcc "/usr/bin/gcc-12" && \
    update-alternatives --set g++ "/usr/bin/g++-12"
RUN echo "alias ll='ls -l'" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
ARG INC_VERSION=a2931eaa4052eec195be3c79a13f7bfa23e54473 # From ww 26
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake==3.22.1 && \
    /opt/conda/bin/conda install -c conda-forge jemalloc==5.2.1 gperftools==2.10 pybind11==2.10.4 && \
    /opt/conda/bin/conda install -c intel mkl==2023.1.0 mkl-include==2023.1.0 intel-openmp==2023.1.0 && \
    /opt/conda/bin/conda clean -ya  && \
    pip3 install --pre torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu && \
    pip install setuptools==58.2.0 && \
    pip install cpuid nltk evaluate protobuf absl-py rouge-score==0.1.2 tqdm numpy cython sentencepiece accelerate && \
    pip install git+https://github.com/intel/neural-compressor.git@${INC_VERSION}


# LLVM build
ENV LD_LIBRARY_PATH "/opt/conda/lib":${LD_LIBRARY_PATH}
#ARG 
FROM conda AS llvm
COPY --from=conda /opt/conda /opt/conda
ENV CONDA_PREFIX "/opt/conda"
WORKDIR /opt/workdir
RUN cd /opt/workdir && \
    export ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") && \
    mkdir llvm-project && cd llvm-project && \
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz && \
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz && \
    tar -xf cmake-16.0.6.src.tar.xz && mv cmake-16.0.6.src cmake && \
    tar -xf llvm-16.0.6.src.tar.xz && mv llvm-16.0.6.src llvm && \
    mkdir build && cd build && \
    export DEB_BUILD_MAINT_OPTIONS=hardening=-format && \
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_BUILD_LLVM_DYLIB=ON   -DLLVM_INCLUDE_BENCHMARKS=OFF ../llvm/ && \
    cmake --build . -j $(nproc) && \
    export LLVM_ROOT=$CONDA_PREFIX && \
    cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX -Wl,-rpath,$CONDA_PREFIX" -P cmake_install.cmake && \
    ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13 && \
    export PATH=${LLVM_ROOT}/bin:$PATH && \
    export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH && \
    export USE_LLVM=${LLVM_ROOT} && \
    export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm


FROM llvm AS build
COPY --from=llvm /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/gptj-99 code/gptj-99
COPY ./code/user_config.py code/user_config.py
ENV CONDA_PREFIX "/opt/conda"
RUN update-ca-certificates -f && \
    cd /opt/workdir/code/gptj-99/pytorch-cpu && mkdir gpt-j-env && cd gpt-j-env && \
    git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu && \
    cd ipex-cpu && git checkout v2.1.0.dev+cpu.llm.mlperf && \
    export IPEX_DIR=${PWD} && \
    git submodule sync && \
    git submodule update --init --recursive && \
    export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1 && \
    python setup.py clean && \
    python setup.py install && \
    unset DNNL_GRAPH_BUILD_COMPILER_BACKEND && \
    unset LLVM_DIR && \
    unset USE_LLVM && \
    python -m pip install transformers==4.28.1 && cd .. && \
    cd /opt/workdir/code/gptj-99/pytorch-cpu/gpt-j-env && \
    git clone https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && \
    export MLPERF_INFERENCE_ROOT=${PWD} && \
    git submodule update --init --recursive third_party/pybind/ && \
    cd loadgen && \
    python -m pip install . && \
    cd /opt/workdir/code/gptj-99/pytorch-cpu/utils && \
    python -m pip install .


#ENV LD_LIBRARY_PATH "/opt/conda/lib/python3.8/site-packages/lib/"
FROM dev-base as mp
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/workdir /opt/workdir
WORKDIR /opt/workdir
ENV CONDA_PREFIX "/opt/conda"
