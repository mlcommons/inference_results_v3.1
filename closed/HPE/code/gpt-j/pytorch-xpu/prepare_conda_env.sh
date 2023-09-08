#!/bin/bash

set -x
# Create new env and activate it
conda install -c intel -y mkl==2023.1.0 mkl-include==2023.1.0
conda install -c conda-forge -y jemalloc==5.2.1
conda install -c conda-forge -y libstdcxx-ng=12 # optional, required only if you meet import error: 'GLIBCXX_x.x.x' not found
pip install torch==2.0.1a0 -f https://developer.intel.com/ipex-whl-stable-xpu
pip install accelerate datasets evaluate nltk rouge_score simplejson

set +x

