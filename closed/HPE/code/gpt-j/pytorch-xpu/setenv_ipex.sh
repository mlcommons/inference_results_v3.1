#!/bin/bash

set -x

export ONEAPI_ROOT=/opt/intel/oneapi
export DPCPPROOT=${ONEAPI_ROOT}/compiler/latest  # 2023.1
source ${DPCPPROOT}/env/vars.sh
export MKLROOT=${ONEAPI_ROOT}/mkl/latest
source ${MKLROOT}/env/vars.sh
export BUILD_SEPARATE_OPS=ON
export USE_XETLA=ON
export BUILD_WITH_CPU=OFF
export USE_AOT_DEVLIST='pvc'

set +x

