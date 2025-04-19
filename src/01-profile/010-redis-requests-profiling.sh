#!/bin/bash

set -v
source config.sh

OUTPUT_DIR=/nfs_homes/dmasouros/project/src/profile/redis/requests/
MODE="remote"
for nreq in 10000; do
    for nclients in 10 100 200 500 1000; do
	for ratio in "1:10"; do
	    ts=$(date +"%s")
	    mkdir -p $OUTPUT_DIR/$ts
	    echo "req,clients,ratio" | tee -a $OUTPUT_DIR/$ts/config.txt
	    echo "$nreq,$nclients,$ratio" | tee -a $OUTPUT_DIR/$ts/config.txt
	    echo "$MODE" | tee -a $OUTPUT_DIR/$ts/config.txt
	    echo "--- Starting redis-server in background---"
	    docker run -d --rm --privileged --cpuset-cpus=0-63 -P -v $OUTPUT_DIR/$ts:/results --name redis-server --network redis-network dmasouros/redis $MODE
	    sleep 2s
	    echo "--- Starting monitoring in background---"
	    python3 -u ${MONITOR_HOME}/watcher.py -c 1000 -s 1000 -o "${OUTPUT_DIR}/$ts/" >> $OUTPUT_DIR/$ts/watcher.log 2>&1 &
	    pid=$!
	    sleep 2s
	    echo "PID=$pid"
	    echo "Starting memtier benchmarking"
            docker run --rm -ti \
                --name=memtier-benchmark \
		--cpuset-cpus=64-127 \
		-v "${OUTPUT_DIR}/$ts":/results \
                --net=redis-network \
		dmasouros/memtier \
                "redis" \
                "$nreq" \
                "$nclients" \
                "$ratio" \
                "$ts.log"
	    sleep 2s
	    echo "Killing monitoring"
	    kill -2 $pid
	    echo "Removing redis-server"
	    docker rm -f redis-server
	    sleep 4s
	    echo "--- SYSTEM MONITORING"
	    cat $OUTPUT_DIR/$ts/sysmon_results_perf.out
	    echo "--- CONTAINER MONITORING"
	    cat $OUTPUT_DIR/$ts/results*out
	    sleep 2s
done; done; done
