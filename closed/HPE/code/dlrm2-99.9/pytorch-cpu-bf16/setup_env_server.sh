
export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
cpu_per_socket=$((number_cores/number_sockets))

export NUM_SOCKETS=$number_sockets        # i.e. 8
export CPUS_PER_SOCKET=$cpu_per_socket   # i.e. 28
export CPUS_PER_PROCESS=$cpu_per_socket  # which determine how much processes will be used

export CPUS_PER_INSTANCE=8  # instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                             # total-instance = instance-per-process * process-per-socket
export CPUS_FOR_LOADGEN=2   # number of cpus for loadgen
                            # finally used in our code is max(CPUS_FOR_LOADGEN, left cores for instances)
export BATCH_SIZE=2048
export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
