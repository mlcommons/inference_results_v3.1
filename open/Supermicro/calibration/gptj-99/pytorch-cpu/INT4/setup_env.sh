CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data

export CALIBRATION_DATA_JSON=${WORKLOAD_DATA}/calibration-data/cnn_dailymail_calibration.json

export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint

export QUANTIZED_MODEL_DIR=${WORKLOAD_DATA}/quantized-int4-model

mkdir -p ${QUANTIZED_MODEL_DIR}
