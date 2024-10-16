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

ARG BENCHMARK_NAME=dlrm-v2-99
ARG IMPL_ID=pytorch-cpu-int8
ARG BASE_IMAGE=rockylinux:8.6
ARG PYTHON_VERSION=3.9
ARG GCC_MAJOR_VERSION=12
ARG IPEX_BRANCH=llm_feature_branch
ARG PYTORCH_WHL=https://download.pytorch.org/whl/nightly/cpu-cxx11-abi/torch-2.1.0.dev20230715%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl

FROM ${BASE_IMAGE} AS dev-base
ARG GCC_MAJOR_VERSION
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    curl \
    vim \
    numactl \
    cmake \
    sudo \
    wget \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc-c++ \
    && rm -rf /var/lib/yum/lists
RUN echo "source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION
ARG PYTORCH_WHL
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -c conda-forge -y \
                                    cmake==3.26.4 \
                                    gperftools==2.10 && \
    /opt/conda/bin/conda install -y mkl==2023.1.0 \
                                    mkl-include \
                                    numpy==1.25 \
                                    ninja==1.10.2 \
                                    pyyaml==6.0 \
                                    setuptools==68.0.0 \
                                    cffi==1.15.1 \
                                    typing_extensions==4.7.1 \
                                    future==0.18.3 \
                                    six==1.16.0 \
                                    requests==2.31.0 \
                                    dataclasses==0.8 \
                                    psutil==5.9.0 \
                                    --no-update-deps && \
    /opt/conda/bin/conda clean -ya
RUN pip install -e git+https://github.com/mlperf/logging@3.0.0-rc2#egg=mlperf-logging && \
    pip install absl-py==1.4.0 \
                tqdm==4.65.0 \
                onnx==1.14.0 \
                lark-parser==0.12.0 \
                hypothesis==6.82.0 \
                ${PYTORCH_WHL} \
                pyre-extensions==0.0.30 \
                scikit-learn==1.3.0

FROM dev-base AS build
ARG GCC_MAJOR_VERSION
ARG IPEX_BRANCH
ARG BENCHMARK_NAME
ARG IMPL_ID
COPY --from=conda /opt/conda /opt/conda
COPY ./code/${BENCHMARK_NAME}/${IMPL_ID}/ipex.patch ipex.patch
COPY ./code/${BENCHMARK_NAME}/${IMPL_ID}/onednngraph.patch onednngraph.patch
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    git clone -b ${IPEX_BRANCH} https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch && \
    cd intel-extension-for-pytorch && \
    git apply ../ipex.patch && \
    git submodule sync && git submodule update --init --recursive && \
    cd third_party/libxsmm && git checkout c21bc5ddb4 && cd ../../ && \
    cd third_party/ideep && rm -rf mkl-dnn && git checkout b5eadff696 && \
    git submodule sync && git submodule update --init --recursive && \
    cd mkl-dnn/ && git apply -p1 ../../../../onednngraph.patch && cd ../../../ && \
    python setup.py install

FROM dev-base as mp
ARG GCC_MAJOR_VERSION
ARG BENCHMARK_NAME
ARG IMPL_ID
COPY --from=build /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK_NAME}/${IMPL_ID} code/${BENCHMARK_NAME}/${IMPL_ID}
COPY ./code/run_clean.sh code/run_clean.sh
COPY ./code/user_config.py code/user_config.py
RUN source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git inference && \
    cd inference && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd .. && cp ./mlperf.conf /opt/workdir/code/${BENCHMARK_NAME}/${IMPL_ID}/.

ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV CONDA_PREFIX "/opt/conda"
