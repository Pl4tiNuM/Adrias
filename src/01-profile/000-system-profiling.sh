#!/bin/bash

THYMESISFLOW_HOME="/nfs_homes/dmasouros/ThymesisFlow/libthymesisflow"
OUTPUT_DIR="system"
mkdir -p ${OUTPUT_DIR}
MODE="remote"
for rsrc in "cpu" "l2" "l3" "memBw" "memCap"
do
    for njobs in 1 2 4 8 16 32
    do
	echo "now running $njobs $rsrc microbenchmarks" | tee -a ${OUTPUT_DIR}/${rsrc}_fpga.log ${OUTPUT_DIR}/${rsrc}_perf.log
	# Monitor socket metrics
        perf stat -e LLC-loads-misses,LLC-store-misses,LLC-loads,LLC-stores,LLC-prefetches,mem-loads,mem-stores -x, -a -A --per-socket -I 1000 -o ${OUTPUT_DIR}/${rsrc}_perf.log --append &
        PERF_PID=$!
	echo "perf PID: $PERF_PID"
	python -u ${THYMESISFLOW_HOME}/scripts/read_counters.py >> ${OUTPUT_DIR}/${rsrc}_fpga.log &
	FPGA_PID=$!
	echo "fpga PID: $FPGA_PID"
    	for (( i = 0; i < njobs; i++ ))
    	do
	    docker run --rm --privileged \
		       --memory=16g \
	               --name "$rsrc-$i" \
	               --cpuset-cpus=0-63 \
	               dmasouros/ibench $MODE $rsrc &
    	done
	echo "started containers... now sleeping zZzz"
    	sleep 30s
        echo "killing containers"
        docker rm -f $(docker ps | awk '{print $NF}' | grep -v NAMES)
        echo "killing perf and read_counters"
        kill -9 $FPGA_PID
        kill -9 $PERF_PID
        sleep 10s
    done
done
