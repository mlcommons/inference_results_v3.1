  export WORKDIR=$PWD
  REPODIR=$WORKDIR/frameworks.ai.benchmarking.mlperf.submission.inference-submission-v3-1

  PATTERN='[-a-zA-Z0-9_]*='
  if [ $# -lt "0" ] ; then
      echo 'ERROR:'
      printf 'Please use following parameters:
      --code=<mlperf workload repo directory>
      '
      exit 1
  fi

  for i in "$@"
  do
      case $i in
         --code=*)
              code=`echo $i | sed "s/${PATTERN}//"`;;
          *)
              echo "Parameter $i not recognized."; exit 1;;
      esac
  done

  if [[ -n $code && -d ${code} ]];then
     REPODIR=$code
  fi

  echo "Install python packages"
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  pip install https://download.pytorch.org/whl/nightly/cpu-cxx11-abi/torch-2.1.0.dev20230715%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl
  pip install pyre-extensions==0.0.30
  pip install scikit-learn==1.3.0

  echo "Install loadgen"
  git clone https://github.com/mlcommons/inference.git
  cd inference
  git log -1
  git submodule update --init --recursive
  cd loadgen
  CFLAGS="-std=c++14" python setup.py install
  cd ..; cp ${WORKDIR}/inference/mlperf.conf ${REPODIR}/closed/Intel/code/dlrm2-99.9/pytorch-cpu-int8/.

  echo "Clone source code and Install"
  echo "Install Intel Extension for PyTorch"
  cd ${WORKDIR}
  # clone Intel Extension for PyTorch
  git clone -b llm_feature_branch https://github.com/intel/intel-extension-for-pytorch.git intel-extension-for-pytorch
  cd intel-extension-for-pytorch
  git apply ${REPODIR}/closed/Intel/code/dlrm2-99.9/pytorch-cpu-int8/ipex.patch
  git submodule sync
  git submodule update --init --recursive
  cd third_party/libxsmm
  git checkout c21bc5ddb4
  cd ../ideep
  rm -rf mkl-dnn
  git checkout b5eadff696
  git submodule sync
  git submodule update --init --recursive
  cd mkl-dnn
  patch -p1 < ${REPODIR}/closed/Intel/code/dlrm2-99.9/pytorch-cpu-int8/onednngraph.patch
  git log -1
  cd ${WORKDIR}/intel-extension-for-pytorch
  python setup.py install
  cd ..
