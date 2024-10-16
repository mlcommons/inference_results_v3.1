#!/bin/bash 
set -ex 
proxy=$1
if [ ! -d  ~/scripts/ ]
then
    mkdir ~/scripts/
fi

#CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )" 

: ${model=${1:-""}}
: ${prepare=${2:-"True"}}
: ${performance=${3:-"True"}}
: ${accuracy=${4:-"True"}}
: ${compliance=${5:-"True"}}
: ${offline=${6:-"True"}}
: ${server=${7:-"True"}}
: ${sensors=${8:-"True"}}
: ${type=${9:-"int8"}}

scenario1="Offline"
scenario2="Server"

cp benchmarking.py dir2tree.py base_info.json ~/scripts/

if [ ${compliance} == "True" ]
    then
        TEST1="TEST01"
        TEST2="TEST05"
        TEST3="TEST04"
fi

#read -p "please enter your host proxy:" http_proxy
#export http_proxy=${proxy}
#export https_proxy=${proxy}


rm -rf /data/mlperf_data/inference
git clone https://github.com/mlcommons/inference.git /data/mlperf_data/inference


case "${model}" in
    "rnnt")
        cd /opt/workdir/code/rnnt/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "resnet50")
        cd /opt/workdir/code/resnet50/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "retinanet")
        cd /opt/workdir/code/retinanet/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "3d-unet-99.9")
        cd /opt/workdir/code/3d-unet-99.9/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "bert-99")
        cd /opt/workdir/code/bert-99/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "dlrm-99.9")
        cd /opt/workdir/code/dlrm/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "gptj-99")
        cd /opt/workdir/code/gptj-99/pytorch-cpu
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;;
    "dlrm-v2-99")
        cd /opt/workdir/code/dlrm-v2-99/pytorch-cpu-${type}
        CUR_DIR=`pwd`
        OUTPUT_DIR=${CUR_DIR}
    ;; 
esac

if [ "${model}" == "" ]
    then
        read -p "which model?(resnet50,retinanet,rnnt,3d-unet-99.9,bert-99,dlrm-99.9,gptj-99,dlrm2-99.9,dlrm-v2-99):" model
fi

if [ ${sensors} == "True" ];then
    rm -f /data/mlperf_data/information_${model}_2.txt
    touch -f /data/mlperf_data/information_${model}_2.txt
    pushd ${OUTPUT_DIR}
    echo "conda list:" >> /data/mlperf_data/information_${model}_2.txt
    conda list >> /data/mlperf_data/information_${model}_2.txt || true
    echo "" >> /data/mlperf_data/information_${model}_2.txt
    popd
fi

if [ ${model} == "resnet50" ]
then
    export DATA_CAL_DIR=calibration_dataset
    export CHECKPOINT=resnet50-fp32-model.pth
    export DATA_DIR=${OUTPUT_DIR}/ILSVRC2012_img_val
    export RN50_START=${OUTPUT_DIR}/models/resnet50-start-int8-model.pth
    export RN50_END=${OUTPUT_DIR}/models/resnet50-end-int8-model.pth
    export RN50_FULL=${OUTPUT_DIR}/models/resnet50-full.pth
fi

if [ ${model} == "retinanet" ] 
then 
    if [ ${prepare} == "True" ]
    then
        cp -r /data/mlperf_data/retinanet/data /opt/workdir/code/retinanet/pytorch-cpu/
    fi
    export WORKLOAD_DATA=/opt/workdir/code/retinanet/pytorch-cpu/data
    export ENV_DEPS_DIR=/opt/workdir/code/retinanet/pytorch-cpu/retinanet-env
    export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
    export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
    export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
    export DATA_DIR=${WORKLOAD_DATA}/openimages
    export MODEL_PATH=${WORKLOAD_DATA}/retinanet-int8-model.pth
fi

if [ ${model} == "rnnt" ]
then
    mkdir -p /opt/workdir/code/rnnt/pytorch-cpu/mlperf-rnnt-librispeech
    cp -f /data/mlperf_data/rnnt/mlperf-rnnt-librispeech/rnnt.pt /opt/workdir/code/rnnt/pytorch-cpu/mlperf-rnnt-librispeech/
fi

if [ ${model} == "3d-unet-99.9" ]
then
    if [ ${prepare} == "True" ]
    then
        mkdir /root/mlperf_data
	cp -r /data/mlperf_data/3dunet-kits /root/mlperf_data
    fi
    OUTPUT_DIR=${CUR_DIR}/output_logs
fi 

if [ ${model} == "bert-99" ]
then
    export DATA_PATH=/data/mlperf_data/bert
    OUTPUT_DIR=${CUR_DIR}/test_log
fi 

if [ ${model} == "dlrm-99.9" ]
then
    if [ ${prepare} == "True" ]
    then
        if [ ! -d /data/mlperf_data/raw_dlrm ] && [ -d /data/mlperf_data/dlrm* ]
        then
            mv /data/mlperf_data/dlrm* /data/mlperf_data/raw_dlrm
        fi
    fi
    mkdir -p /data/mlperf_data/dlrm
    export MODEL=/data/mlperf_data/raw_dlrm/
    export DATASET=/data/mlperf_data/raw_dlrm/
    export DUMP_PATH=/data/mlperf_data/dlrm
    export MODEL_DIR=/data/mlperf_data/dlrm
    export DATA_DIR=/data/mlperf_data/dlrm
fi

if [ ${model} == "gptj-99" ]
then
    if [ ${prepare} == "True" ]
    then

        if [ ! -d /data/mlperf_data/gpt-j/data ]
        then
            echo 'warning: data not found'
        fi
	if [ -d /opt/workdir/code/${model}/pytorch-cpu/data ] || [ -L /opt/workdir/code/${model}/pytorch-cpu/data ] 
	then
            rm -rf /opt/workdir/code/${model}/pytorch-cpu/data
        fi
        ln -s /data/mlperf_data/gpt-j/data /opt/workdir/code/${model}/pytorch-cpu/data
    fi
    CUR_DIR=$(pwd)
    export WORKLOAD_DATA=${CUR_DIR}/data
    export CALIBRATION_DATA_JSON=${WORKLOAD_DATA}/calibration-data/cnn_dailymail_calibration.json
    export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
    export VALIDATION_DATA_JSON=${WORKLOAD_DATA}/validation-data/cnn_dailymail_validation.json
    export INT8_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int8-model
    export INT4_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int4-model
    export INT4_CALIBRATION_DIR=${WORKLOAD_DATA}/quantized-int4-model

    mkdir -p ${INT8_MODEL_DIR}
    mkdir -p ${INT4_MODEL_DIR}
fi 

if [ ${model} == "dlrm-v2-99" ]
then
    export DATA_DIR=/data/mlperf_data/dlrm_2/data_npy/
    export MODEL_DIR=/data/mlperf_data/dlrm_2/model/${type}/
    export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
    number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
    cpu_per_socket=$((number_cores/number_sockets))
    export NUM_SOCKETS=$number_sockets        # i.e. 8
    export CPUS_PER_SOCKET=$cpu_per_socket   # i.e. 28
    export CPUS_PER_PROCESS=$cpu_per_socket  # which determine how much processes will be used
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    if [ ${type}  == "int8" ]
    then
        ln -sf ${MODEL_DIR}/dlrm-multihot-pytorch_int8.pt dlrm_int8.pt
    fi
fi 


if [ ${prepare} == "True" ]
then
    python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} --prepare --datatype $type
fi



if [ ${accuracy} == "True" ]
then

    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Offline/accuracy/Offline_original_true_BS256_28_4_SL2
    fi

    if [ ${model} == "dlrm-99.9" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/AccuracyOnly/Offline
    fi

    if [ ${model} == "dlrm-v2-99" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Offline/accuracy
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=4 
            export CPUS_FOR_LOADGEN=4 
            export BATCH_SIZE=4096
        fi
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=1
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=500
        fi
    fi

    if [ ${offline} == "True" ]
    then
        sleep 120 
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} --accuracy --datatype $type
    fi
    
    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Server/accuracy/Server_original_true_PBS4_16_1_BS128_12_8_SL8_RSP9_QOS233500
    fi

    if [ ${model} == "dlrm-99.9" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/AccuracyOnly/Server
    fi

    if [ ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Server/accuracy/
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=8
            export CPUS_FOR_LOADGEN=2
            export BATCH_SIZE=2048
        fi 
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=2
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=400
        fi
    fi

    if [ ${server} == "True" ]
    then
        if [ ${model} != "3d-unet-99.9" ];then
            sleep 120
            python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario2} -O ${OUTPUT_DIR} --accuracy --datatype $type
        fi
    fi
fi

if [ ${performance} == "True" ]
then
    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Offline/performance/run_1/Offline_original_true_BS256_28_4_SL2
    fi

    if [ ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/PerformanceOnly/Offline
        echo $MODEL_DIR
        echo $DATA_DIR
    fi

    if [ ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Offline/performance/run_1/
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=4 
            export CPUS_FOR_LOADGEN=4 
            export BATCH_SIZE=4096
        fi
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=1
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=500
        fi
    fi

    if [ ${offline} == "True" ]
    then
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} --performance --datatype $type
    fi

    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Server/performance/run_1/Server_original_true_PBS4_16_1_BS128_12_8_SL8_RSP9_QOS233500
    fi

    if [ ${model} == "dlrm-99.9" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/PerformanceOnly/Server
    fi

    if [ ${model} == "dlrm2-99.9" || ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Server/performance/run_1/
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=8
            export CPUS_FOR_LOADGEN=2
            export BATCH_SIZE=2048
        fi 
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=2
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=400
        fi
    fi
    
    if [ ${server} == "True" ]
    then
        if [ ${model} != "3d-unet-99.9" ];then
            sleep 120
            python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario2} -O ${OUTPUT_DIR} --performance --datatype $type
        fi
    fi
fi


if [[ "${compliance}" == "True" && "${mode}" != "gptj-99" ]]
then

    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Offline/performance/run_1/Offline_original_true_BS256_28_4_SL2
    fi

    if [ ${model} == "dlrm-99.9" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/PerformanceOnly/Offline
        echo $MODEL_DIR
        echo $DATA_DIR
    fi

    if [ ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Offline/performance/run_1/
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=4 
            export CPUS_FOR_LOADGEN=4 
            export BATCH_SIZE=4096
        fi
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=1
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=500
        fi
    fi
    

    if [ ${offline} == "True" ]
    then
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} -T ${TEST1} --compliance --datatype $type
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} -T ${TEST2} --compliance --datatype $type
    fi

    if [ ${model} == "resnet50" ]
    then
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario1} -O ${OUTPUT_DIR} -T ${TEST3} --compliance --datatype $type
    fi

    if [ ${model} == "rnnt" ]
    then
        OUTPUT_DIR=${CUR_DIR}/logs/Server/performance/run_1/Server_original_true_PBS4_16_1_BS128_12_8_SL8_RSP9_QOS233500
    fi

    if [ ${model} == "dlrm-99.9" ]
    then
        OUTPUT_DIR=${CUR_DIR}/output/PerformanceOnly/Server
    fi

    if [ ${model} == "dlrm-v2-99"]
    then
        OUTPUT_DIR=${CUR_DIR}/output/pytorch-cpu/dlrm/Server/performance/run_1/
        if [ ${type} == "bf16" ]
        then
            export CPUS_PER_INSTANCE=8
            export CPUS_FOR_LOADGEN=2
            export BATCH_SIZE=2048
        fi 
        if [ ${type} == "int8" ]
        then
            export CPUS_PER_INSTANCE=2
            export CPUS_FOR_LOADGEN=1
            export BATCH_SIZE=400
        fi
    fi

    if [ ${server} == "True" ]
    then
        if [ ${model} != "3d-unet-99.9" ];then
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario2} -O ${OUTPUT_DIR} -T ${TEST1} --compliance --datatype $type
        sleep 120
        python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario2} -O ${OUTPUT_DIR} -T ${TEST2} --compliance --datatype $type
        if [ ${model} == "resnet50" ]
        then
            sleep 120
            python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario2} -O ${OUTPUT_DIR} -T ${TEST3} --compliance --datatype $type
        fi
        fi
    fi
fi

#if [ ${check} == "True" ]
#then
#    python3 ~/scripts/benchmarking.py -B ${CUR_DIR} -m ${model} -s ${scenario} -O ${OUTPUT_DIR} --check
#fi

echo "log position:/data/mlperf_data/results_v3.1"

