CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data

export CALIBRATION_DATA_JSON=${WORKLOAD_DATA}/calibration-data/cnn_dailymail_calibration.json

export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint

export VALIDATION_DATA_JSON=${WORKLOAD_DATA}/validation-data/cnn_dailymail_validation.json

export INT8_MODEL_DIR=${WORKLOAD_DATA}/gpt-j-int8-model
