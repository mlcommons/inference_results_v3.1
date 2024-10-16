set -x

model=$1
dataset=$2
postfix=$3

if [ ${model} == "dlrm2-99.9" ];then
        read -p "Data type?(int8 or bf16):" type
        if [ ${type} != "int8" ] && [ ${type} != "bf16" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'bf16'."
                exit 1
        fi
fi

if [ ${model} == "gptj-99" ];then
        read -p "Data type?(int8 or int4):" type
        if [ ${type} != "int8" ] && [ ${type} != "int4" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'int4'."
                exit 1
        fi
fi

if [ ${model} == "resnet50" ];then
        read -p "Data type?(int8 or int4):" type
        if [ ${type} != "int8" ] && [ ${type} != "int4" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'int4'."
                exit 1
        fi
fi

if [ ${model} == "3d-unet-99.9" ];then
        read -p "Data type?(int8 or int4):" type
        if [ ${type} != "int8" ] && [ ${type} != "int4" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'int4'."
                exit 1
        fi
fi

if [ ${model} == "retinanet" ];then
        read -p "Data type?(int8 or int4):" type
        if [ ${type} != "int8" ] && [ ${type} != "int4" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'int4'."
                exit 1
        fi
fi

if [ ${model} == "bert-99" ];then
        read -p "Data type?(int8 or int4):" type
        if [ ${type} != "int8" ] && [ ${type} != "int4" ];then
                echo "Error: Invalid data type. Please choose 'int8' or 'int4'."
                exit 1
        fi
fi


: ${PerformanceOnly=${4:-"False"}}
: ${AccuracyOnly=${5:-"False"}}
: ${OfflineOnly=${6:-"False"}}
: ${ServerOnly=${7:-"False"}}
: ${ComplianceOnly=${8:-"False"}}
: ${Skip_docker_build=${9:-"False"}}
: ${Skip_data_proprocess=${10:-"False"}}

prepare="True"
performance="True"
accuracy="True"
compliance="True"
offline="True"
server="True"
sensors="True"

if [ ${PerformanceOnly} == "True" ];then
        accuracy="False"
        compliance="False"
fi

if [ ${AccuracyOnly} == "True" ];then
        performance="False"
        compliance="False"
fi

if [ ${OfflineOnly} == "True" ];then
        server="False"
        compliance="False"
fi
if [ ${ServerOnly} == "True" ];then
        offline="False"
        compliance="False"
fi

if [ ${ComplianceOnly} == "True" ];then
        performance="False"
        accuracy="False"
fi

if [ ${Skip_data_proprocess} == "True" ];then
        prepare="False"
fi

if [ ${model} == "dlrm2-99.9" ];then
        BENCHMARK_DIR=$(pwd |awk -F'closed' '{print $1}')
        AUTOMATION_DIR=${BENCHMARK_DIR}/closed/Intel/code/${model}/pytorch-cpu-${type}/automation
        MEASUREMENT_DIR=${BENCHMARK_DIR}/closed/Intel/measurements/1-node-2S-SPR-PyTorch-$(echo $type | tr '[:lower:]' '[:upper:]')/${model}/Offline
        WORKLOAD_DIR=${BENCHMARK_DIR}/closed/Intel/code/${model}/pytorch-cpu-${type}/
else
        BENCHMARK_DIR=$(pwd |awk -F'closed' '{print $1}')
        AUTOMATION_DIR=${BENCHMARK_DIR}/closed/Intel/code/${model}/pytorch-cpu/automation
	MEASUREMENT_DIR=${BENCHMARK_DIR}/closed/Intel/measurements/1-node-2S-SPR-PyTorch-$(echo $type | tr '[:lower:]' '[:upper:]')/${model}/Offline
        WORKLOAD_DIR=${BENCHMARK_DIR}/closed/Intel/code/${model}/pytorch-cpu/
fi

if [ ${Skip_docker_build} == "False" ];then

        if [ -d "${AUTOMATION_DIR}" ];then
                rm -rf ${AUTOMATION_DIR} 
        fi
        if [ -d "${AUTOMATION_DIR}/automation" ];then
                rm -rf ${AUTOMATION_DIR}/automation
        fi

        cp -rf ${BENCHMARK_DIR}/closed/Intel/code/automation ${AUTOMATION_DIR}
        cp -rf ${BENCHMARK_DIR}/closed/Intel/code/automation ${dataset}/.

        cd ${WORKLOAD_DIR}/docker

        #if [ ${model} == "dlrm-99.9" ];then
         #       rm ../../l_HPCKit_p_2022.2.0.191.sh*
         #      wget -w 60 https://registrationcenter-download.intel.com/akdlm/irc_nas/18679/l_HPCKit_p_2022.2.0.191.sh -O ../../l_HPCKit_p_2022.2.0.191.sh
        #fi

        bash ./build_*${type}_container.sh >build.log

        declare -A map=(
        ["dlrm-99.9"]="mlperf_inference_dlrm"
        ["bert-99"]="mlperf_inference_bert"
        ["rnnt"]="mlperf_inference_rnnt"
        ["3d-unet-99.9"]="mlperf_inference_3dunet"
        ["resnet50"]="mlperf_inference_datacenter_resnet50"
        ["retinanet"]="mlperf_inference_datacenter_retinanet"
        ["gptj-99-int8"]="mlperf_inference_gptj"
        ["gptj-99-int4"]="mlperf_inference_gptj_int4"
        ["dlrm2-99.9-int8"]="mlperf_inference_dlrm2"
        ["dlrm2-99.9-bf16"]="mlperf_inference_dlrm2_bf16"
        )

        if [ "${type}" != "" ];then
            image_id=${map["${model}-${type}"]}
        else
            image_id=${map["$model"]}
        fi
        container_id=$(docker run --name intel_${model}_${postfix} --privileged -itd -v ${dataset}:/data/mlperf_data --net=host --ipc=host ${image_id}:3.1 | tail -1)
fi

if [ ${sensors} == "True" ];then
        rm -f ${dataset}/information_${model}.txt
        touch -f ${dataset}/information_${model}.txt
        pushd ${BENCHMARK_DIR}
        echo " git log --oneline | head -n 1:" >> ${dataset}/information_${model}.txt
        git log --oneline | head -n 1 >> ${dataset}/information_${model}.txt || true
        echo "" >> ${dataset}/information_${model}.txt
        popd
        pushd ${dataset}
        echo "who:" >> information_${model}.txt
        who >> information_${model}.txt || true
        echo "" >> information_${model}.txt
        echo "free -h >> information_${model}.txt:" >> information_${model}.txt
        free -h >> information_${model}.txt || true
        echo "" >> information_${model}.txt
        echo "ps -ef | grep python:" >> information_${model}.txt
        ps -ef | grep python >> information_${model}.txt || true
        echo "" >> information_${model}.txt
        echo "lscpu:" >> information_${model}.txt
        lscpu >> information_${model}.txt || true
        dmesg | grep "cpu clock throttled" >> information_${model}.txt || true
        echo "" >> information_${model}.txt
        popd
fi        

if [ ${Skip_docker_build} == "True" ];then
        read -p "please tap docker container_id:" container_id
fi

cp -rf ${BENCHMARK_DIR}/closed/Intel/code/automation ${AUTOMATION_DIR}
cp -rf ${BENCHMARK_DIR}/closed/Intel/code/automation ${dataset}/.
cp -rf ${BENCHMARK_DIR}/closed/Intel/code/run_clean.sh ${dataset}/.

if [ ${model} == "dlrm2-99.9" ];then
        docker exec $container_id sh -c "export http_proxy=$http_proxy && export https_proxy=$https_proxy && rm -rf /opt/workdir/code/${model}/pytorch-cpu-${type}/automation && cp -rf /data/mlperf_data/automation /opt/workdir/code/${model}/pytorch-cpu-${type}/automation && cd /opt/workdir/code/${model}/pytorch-cpu-${type}/automation && prepare=${prepare} performance=${performance} accuracy=${accuracy} compliance=${compliance} sensors=${sensors} model=${model} offline=${offline} server=${server} type=${type} bash benchmarking.sh"
else
        docker exec $container_id sh -c "export http_proxy=$http_proxy && export https_proxy=$https_proxy && rm -rf /opt/workdir/code/${model}/pytorch-cpu/automation && cp -rf /data/mlperf_data/automation /opt/workdir/code/${model}/pytorch-cpu/automation && cd /opt/workdir/code/${model}/pytorch-cpu/automation && prepare=${prepare} performance=${performance} accuracy=${accuracy} compliance=${compliance} sensors=${sensors} model=${model} offline=${offline} server=${server} type=${type} bash benchmarking.sh"
fi
