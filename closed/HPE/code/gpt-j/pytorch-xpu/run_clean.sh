#!/bin/bash

echo '==> Setting CPU configs'
# echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
# sudo echo 0 > /proc/sys/kernel/numa_balancing
# sudo echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo CPU Frequency Mode now is `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq`

echo '==> Setting GPU configs'
for f in $(ls /sys/class/drm/card*/subsystem/card*/gt/gt*/rps_min_freq_mhz); do echo 1600 > ${f}; done
for f in $(ls /sys/class/drm/card*/subsystem/card*/gt/gt*/rps_max_freq_mhz); do echo 1600 > ${f}; done
# to check GPU frequency
echo 'GPU Max Frequency is:'
cat /sys/class/drm/card0/subsystem/card*/gt/gt*/rps_max_freq_mhz
echo 'GPU Min Frequency is:'
cat /sys/class/drm/card0/subsystem/card*/gt/gt*/rps_min_freq_mhz

echo '==> Clean Resources'
echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo 1 > /proc/sys/vm/compact_memory; sleep 1
echo 3 > /proc/sys/vm/drop_caches; sleep 1
