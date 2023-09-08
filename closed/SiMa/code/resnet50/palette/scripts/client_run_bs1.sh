NTP_SERVER=time.google.com
# NTP_SERVER=192.168.1.10
POWER_SERVER=192.168.89.82
# POWER_SERVER=192.168.1.10

/mnt/power-dev/ptd_client_server/client.py \
    --addr $POWER_SERVER \
    --output "./" \
    --run-workload "./run_bs1.sh" \
    --loadgen-logs "loadgen_out" \
    --label "singlestream" \
    --ntp $NTP_SERVER


