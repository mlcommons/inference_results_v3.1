cd /work

echo $MLPERF_SCRATCH_PATH  # Make sure that the container has the MLPERF_SCRATCH_PATH set correctly
ls -al $MLPERF_SCRATCH_PATH  # Make sure that the container mounted the scratch space correctly

#!!!CAUTION, this will delete previous performance logs!!!
#make clean  # Make sure that the build/ directory isn't dirty
#!!!CAUTION, this will delete previous performance logs!!!

make link_dirs  # Link the build/ directory to the scratch space
ls -al build/

#Note: You should have already added custom systems at this point
# refer to READEME.md on adding systems using this script:
#python3 -m scripts.custom_systems.add_custom_system

#make clone_loadgen #sometimes needed?
make build

#for gptj
#make sure to remove -a=90 flags from Makefile.build to enable support for SM80 and SM89
#rm -rf build/TRTLLM #remove if exists
make build_trt_llm
BUILD_TRTLLM=1 make build_harness
#then do make generate_engines ... commands

