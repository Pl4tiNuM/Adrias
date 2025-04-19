#!/bin/bash

set -v
source config.sh

OUTPUT_DIR=/nfs_homes/dmasouros/project/src/profile/redis/interference/
MODE="remote"
nreq=5000
nclients=100
ratio="1:10"
for MODE in "local" "remote"; do
for rsrc in "cpu" "l2" "l3" "memBw" "memCap"; do
    for njobs in 1 2 4 8 16; do
	ts=$(date +"%s")
	mkdir -p $OUTPUT_DIR/$ts
	echo "req,clients,ratio" | tee -a $OUTPUT_DIR/$ts/config.txt
	echo "$nreq,$nclients,$ratio" | tee -a $OUTPUT_DIR/$ts/config.txt
	echo "$MODE" | tee -a $OUTPUT_DIR/$ts/config.txt
	echo "$rsrc,$njobs" | tee -a $OUTPUT_DIR/$ts/config.txt
	echo "--- Starting redis-server in background---"
	docker run -d --rm --privileged --cpuset-cpus=0-63 -P -v $OUTPUT_DIR/$ts:/results --name redis-server --network redis-network dmasouros/redis $MODE
	sleep 1s
	echo "Starting interference microbenchmarks"
  	for (( i = 0; i < njobs; i++ ))
    	do
	    echo "$rsrc - $i"
	    docker run --rm --privileged \
		       --memory=16g \
	               --name "$rsrc-$i" \
	               --cpuset-cpus=0-63 \
	               dmasouros/ibench $MODE $rsrc &
    	done
        echo "--- Starting monitoring in background---"
        python3 -u ${MONITOR_HOME}/watcher.py -c 100 -s 100 -o "${OUTPUT_DIR}/$ts/" > /dev/null 2>&1 &
        pid=$!
	echo "PID=$pid"
	sleep 1s	
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
	echo "Killing rest of containers"
	docker rm -f $(docker ps | awk '{print $NF}' | grep -v NAMES)
	sleep 4s
done; done; done
