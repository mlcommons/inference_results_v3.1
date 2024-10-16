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

ARG BASE_IMAGE=rockylinux:8.7
ARG PYTHON_VERSION=3.9
ARG GCC_MAJOR_VERSION=11
ARG ONEAPI_VERSION=2023.1.0
ARG TORCH_VERSION=2.0.1
ARG IPEX_VERSION=2.0.100
ARG BENCHMARK_NAME=dlrm2-99.9
ARG IMPL_ID=pytorch-cpu-bf16

FROM ${BASE_IMAGE} AS dev-base
ARG GCC_MAJOR_VERSION
ARG ONEAPI_VERSION
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
	DEBIAN_FRONTEND=noninteractive dnf install -y \
	ca-certificates \
	git \
	curl \
	numactl \
	cmake \
	sudo \
	wget \
	procps \
	numactl-devel.x86_64 \
	gcc \
	gcc-c++ \
	gcc-toolset-${GCC_MAJOR_VERSION}-gcc \
	gcc-toolset-${GCC_MAJOR_VERSION}-gcc-c++ \
	&& rm -rf /var/lib/yum/lists
#RUN echo "source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable" >> /root/.bashrc && \
#	echo "source /opt/intel/oneapi/compiler/${ONEAPI_VERSION}/env/vars.sh" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION
ARG ONEAPI_VERSION
ARG TORCH_VERSION
ARG IPEX_VERSION
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
	chmod +x ~/miniconda.sh && \
	~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh && \
	/opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
	/opt/conda/bin/conda install -c conda-forge -y \
		cmake==3.26.4 \
		gperftools==2.10 && \
	/opt/conda/bin/conda install -y \
		intel-openmp==${ONEAPI_VERSION} \
		mkl==${ONEAPI_VERSION} \
		mkl-include==${ONEAPI_VERSION} \
		numpy==1.25.0 \
		--no-update-deps && \
	/opt/conda/bin/conda clean -ya && \
	pip install -e git+https://github.com/mlperf/logging@1.1.0-rc3#egg=mlperf-logging && \
 	pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu && \
	pip install intel-extension-for-pytorch==${IPEX_VERSION} && \
	pip install absl-py==1.4.0 \
				tqdm==4.65.0 \ 
  				torchrec==0.4.0 \
				pyre-extensions==0.0.30 \
				scikit-learn==1.3.0 && \
	pip install fbgemm-gpu-cpu==0.4.1

FROM conda AS build
ARG GCC_MAJOR_VERSION
ARG BENCHMARK_NAME
ARG IMPL_ID
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt/workdir
COPY ./code/${BENCHMARK_NAME}/${IMPL_ID} code/${BENCHMARK_NAME}/${IMPL_ID}
COPY ./code/run_clean.sh code/run_clean.sh
COPY ./code/user_config.py code/user_config.py
RUN source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
	cd /opt/workdir/code/${BENCHMARK_NAME} && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git inference && \
    cd inference && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cd .. && cp ./mlperf.conf /opt/workdir/code/${BENCHMARK_NAME}/${IMPL_ID}/.

ENV CONDA_PREFIX "/opt/conda"