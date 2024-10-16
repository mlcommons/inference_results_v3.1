#!/usr/bin/env bash

sleep 75

pushd /mnt/mlperf_august
# echo power_begin $(date +"%m-%d-%Y %T.%3N") | tee dummy-loadgen-logs/mlperf_log_detail.txt

# export LD_LIBRARY_PATH=/mnt/mlperf_v2/libs
# export MLA_OCM=max

mla-rt -d /lib/firmware/mla_driver.bin /usr/bin/init_mla_mem.lm
export LD_LIBRARY_PATH=libs_argmax

./run2 -t 0 -i 24:3:224:224 -o 24:64 -r 0 -b 24 -s 2 -m false -d /mnt/mlperf_sr/mlperf_resnet50_dataset.dat -l /mnt/mlperf_august/libs_argmax -a /mnt/mlperf_august/models_final


# ./run -t 0 -i "1:3:224:224" -o "1:1001" -r 0 -b 1
# echo power_end $(date +"%m-%d-%Y %T.%3N") | tee -a dummy-loadgen-logs/mlperf_log_detail.txt

popd

mkdir -p loadgen_out/
mv /mnt/mlperf_august/mlperf_log_detail.txt loadgen_out/
mv /mnt/mlperf_august/mlperf_log_summary.txt loadgen_out/
mv /mnt/mlperf_august/mlperf_log_trace.json loadgen_out/
mv /mnt/mlperf_august/mlperf_log_accuracy.json loadgen_out/

