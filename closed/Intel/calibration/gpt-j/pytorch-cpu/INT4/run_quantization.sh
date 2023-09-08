
CAL_SAMPLES=(128)
GROUPSIZE=(128)
COMPRESSION_FACTOR=2
for cal_size in ${CAL_SAMPLES[@]}; do
    for g in ${GROUPSIZE[@]}; do
		echo "Running groups ${g} and samples ${cal_size}"
		numactl -m 0 -C 0-55 python -u gptj.py --model ${CHECKPOINT_DIR} \
	  --wbits 4 \
	  --true-sequential \
	  --act-order \
	  --groupsize ${g} \
	  --save ${QUANTIZED_MODEL_DIR}/gpt-j-quantized_model_${g}g_${cal_size}samples.pt \
	  --calib-data-path ${CALIBRATION_DATA_JSON} \
	  --nsamples ${cal_size} \
	  --quant-config-output ${QUANTIZED_MODEL_DIR}/gpt-j-quantized_model_params.json \
	  --compression-factor ${COMPRESSION_FACTOR} \
	  --compression-dim "N" \
	  --calib-iters ${cal_size} \
	  --quantize-lm-head \
	  2>&1 | tee log_${g}groups_${cal_size}samples_cf_${COMPRESSION_FACTOR}.log

    done
done
