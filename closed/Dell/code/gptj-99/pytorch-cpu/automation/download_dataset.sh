#!/bin/bash 
set -ex
model=$1
path=$2
conda_path=$3
if [ "${model}" == "" ]
    then
        read -p "which model?(resnet50,retinanet,rnnt,3d-unet,bert,gptj-99,dlrm2-99.9,all(enter all for all the model)):" model
fi
echo $model
if [ "${path}" == "" ]
    then
        read -p "The path to put the data(for example:/data/mlperf_data):" path
fi
echo $path
if [ "${conda_path}" == "" ]
    then
	read -p "The path to conda (for example the path to miniconda3/bin):" conda_path
fi
echo ${conda_path}


#read -p "The path to put the data(for example:~/inference-submission-v3-0):" benchmark 
#echo $benchmark

WORK_DIR=`pwd`
if [ ${model} == "all" ]
then
    rm -rf ${path}
fi

if [ ! -d ${path} ]
then
    mkdir -p ${path}
fi

#cp -r `pwd` ${path}

if [ ${model} == "resnet50" ] || [ ${model} == "all" ]
then
    #resnet50:
    echo 'downloading '${model}' dataset/model in '${path}
    pushd ${path}
    if [ -d ./resnet50 ];then
      rm -rf resnet50 
    fi
    mkdir -p resnet50
    cd resnet50
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    if [ -d "ILSVRC2012_img_val" ]; then
        rm -r ILSVRC2012_img_val
    fi
    mkdir ILSVRC2012_img_val
    tar -xvf ILSVRC2012_img_val.tar -C ILSVRC2012_img_val
    #cp val_data/*.txt ILSVRC2012_img_val/
    wget --no-check-certificate https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth -O resnet50.pth
    mv resnet50.pth resnet50-fp32-model.pth
    popd
    echo ${model}' dataset/model download finished in '${path}
fi

if [ ${model} == "retinanet" ] || [ ${model} == "all" ]
then
    #retinanet
    echo 'downloading '${model}' dataset/model in '${path}
    pushd ${path}
    if [ -d ./retinanet ];then
       rm -rf retinanet
    fi
    mkdir -p retinanet
    cd retinanet
    mkdir -p data
    CUR_DIR=$(pwd)
    export WORKLOAD_DATA=${CUR_DIR}/data
    mkdir -p ${WORKLOAD_DATA}
    export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env
    cd ${WORKLOAD_DATA}
    wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
    popd
    echo ${model}' dataset/model download finished in '${path}
fi

if [ ${model} == "rnnt" ] || [ ${model} == "all" ]
then
    #rnnt:
    echo 'downloading '${model}' dataset/model in '${path}
    pushd ${path}
    if [ -d ./rnnt ];then
       rm -rf rnnt
    fi
    mkdir -p rnnt
    cd rnnt
    mkdir -p mlperf-rnnt-librispeech/local_data
    cd mlperf-rnnt-librispeech
    wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O ./rnnt.pt
    popd
    echo ${model}' dataset/model download finished in '${path}
fi 

if [ ${model} == "3d-unet" ] || [ ${model} == "all" ]
then
    #3dunet:
    echo 'downloading '${model}' dataset/model in '${path}
    pushd ${path}
    if [ -d ./3dunet-kits ];then
       rm -rf 3dunet-kits
    fi
    mkdir -p 3dunet-kits
    cd 3dunet-kits
    rm -rf kits19
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    popd
    echo ${model}' dataset/model download finished in '${path}
fi

if [ ${model} == "bert" ] || [ ${model} == "all" ]
then
    #bert:
    echo 'downloading '${model}' dataset/model in '${path} 
    pushd ${path}
    if [ -d ./bert ];then
       rm -rf bert
    fi
    mkdir -p bert/{dataset,model}
    cd bert
    rm -rf ./dataset/dev-v1.1.json
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O ./dataset/dev-v1.1.json
    rm -rf model
    git clone https://huggingface.co/bert-large-uncased model
    cd model
    wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
    popd
    echo ${model}' dataset/model download finished in '${path} 
fi

if [ ${model} == "gptj-99" ] || [ ${model} == "all" ]
then
    #gptj-99:
    echo 'downloading '${model}' dataset/model in '${path} 
    pushd ${path}
    if [ -d ./gpt-j ];then
       rm -rf gpt-j
    fi
    mkdir -p gpt-j/data/gpt-j-checkpoint
    cd gpt-j
    wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
    unzip gpt-j-checkpoint.zip
    mv gpt-j/checkpoint-final/* ./data/gpt-j-checkpoint/
    popd
    echo ${model}' dataset/model download finished in '${path} 

    export PATH=${conda_path}:$PATH

    if [ -d ../../calibration/gpt-j/pytorch-cpu/INT8 ];then
        echo "do int8 calibration"
        pushd ../../calibration/gpt-j/pytorch-cpu/INT8/
        source /opt/rh/gcc-toolset-11/enable
        bash ./prepare_calibration_env.sh gpt-j-calibration-int8-env
        source activate gpt-j-calibration-int8-env
        
        CUR_DIR=$(pwd)
        export WORKLOAD_DATA=${CUR_DIR}/data
        if [ -d "./data" ] || [ -L "./data" ];then
            rm -rf data
        fi
        ln -s $path/gpt-j/data data

	if [ ! -d ${path}/gpt-j/data/calibration-data ];then 
            python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${path}/gpt-j/data/calibration-data
        fi
	if [ ! -d ${path}/${path}/gpt-j/data/calibration-data ];then
            python ../../../../code/gptj-99/pytorch-cpu/download-dataset.py --split validation --output-dir ${path}/gpt-j/data/validation-data
        fi
 
        CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
        source setup_env.sh
        bash run_quantization.sh
        source deactivate
        popd
    fi

    if [ -d ../../calibration/gpt-j/pytorch-cpu/INT4 ];then
        echo "do int4 calibration"
        pushd ../../calibration/gpt-j/pytorch-cpu/INT4/
        conda env create -f quantization-env.yaml
        if [ -d "./data" ] || [ -L "./data" ];then
            rm -rf data
        fi
        ln -s $path/gpt-j/data data

        source activate gpt-j-calibration-int4-env

        if [ ! -d ${path}/gpt-j/data/calibration-data ];then 
            python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${path}/gpt-j/data/calibration-data
        fi
        if [ ! -d ${path}/${path}/gpt-j/data/calibration-data ];then
            python ../../../../code/gptj-99/pytorch-cpu/download-dataset.py --split validation --output-dir ${path}/gpt-j/data/validation-data
        fi
        source setup_env.sh
        bash run_calibration_int4.sh
        source deactivate
        popd
    fi
fi

if [ ${model} == "dlrm2-99.9" ] || [ ${model} == "all" ]
then
    #bert:
    echo "The data set and model of dlrm2 need to be downloaded by yourself.\
          For the download method, refer to the README of dlrm2's code.\
          Please put the data into <data_path>/dlrm_2/data_npy/,\
          put the model into <data_path>/dlrm_2/model/{int8,bf16}/"
fi


if [ ${model} != "resnet50" ] && [ ${model} != "retinanet"  ] && [ ${model} != "rnnt"  ] && [ ${model} != "3d-unet"  ] && [ ${model} != "bert"  ] && [ ${model} != "gptj-99"  ] && [ ${model} != "dlrm2-99.9" ] && [ ${model} != "all"  ]
then
    echo 'please enter the right model (resnet50,retinanet,rnnt,3d-unet,bert,gptj-99,dlrm2-99.9,all) and path'
fi
