#!/bin/bash

set -v
source config.sh

SPARK_BENCH=$1

mkdir -p "/nfs_homes/dmasouros/project/new_results/$SPARK_BENCH"
# OUTPUT_DIR=/nfs_homes/dmasouros/project/new_results/$SPARK_BENCH

# for MODE in "local" "remote"; do
# for rsrc in "cpu" "l2" "l3" "memBw"; do
# for njobs in 1 2 4 8 16; do

for MODE in "remote"; do
for rsrc in "cpu"; do
for njobs in 1; do
	mkdir -p "/nfs_homes/dmasouros/project/new_results/$SPARK_BENCH/$MODE/$rsrc/$njobs"
	OUTPUT_DIR=/nfs_homes/dmasouros/project/new_results/$SPARK_BENCH/$MODE/$rsrc/$njobs/
	
	ts=$(date +"%s")
	echo "--- Initializing Spark prerequisites ---"
    docker run -d -P --rm --privileged --cpuset-cpus=0-63 \
            -v ${OUTPUT_DIR}:/results \
            --name "tmp-$SPARK_BENCH-$ts" \
            dmasouros/spark $MODE >> ${OUTPUT_DIR}/$ts.log 2>&1
	sleep 1s
    SYSTEM_UP=""
    while [ -z "${SYSTEM_UP}" ]; do
	    SYSTEM_UP=$(docker logs "tmp-$SPARK_BENCH-$ts" 2>&1 | grep "Running...")
	    sleep 1s
    done

    # Change dataset size to small and create it
    docker exec "tmp-$SPARK_BENCH-$ts" sed -i 's/tiny/small/g' /HiBench/conf/hibench.conf >> ${OUTPUT_DIR}/$ts.log 2>&1
    docker exec "tmp-$SPARK_BENCH-$ts" /HiBench/bin/workloads/${S["$SPARK_BENCH"]}/$SPARK_BENCH/prepare/prepare.sh >> ${OUTPUT_DIR}/$ts.log 2>&1

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
        python3 -u ${MONITOR_HOME}/watcher.py -c 100 -s 100 -o "${OUTPUT_DIR}" > /dev/null 2>&1 &
        pid=$!
	echo "PID=$pid"
	sleep 1s	
	echo "Starting spark benchmarking"
    # Rename container to start monitoring
    docker rename "tmp-$SPARK_BENCH-$ts" "spark-$SPARK_BENCH-$ts" >> ${OUTPUT_DIR}/$ts.log 2>&1
    sleep 2s
    # Run benchmark
    docker exec "tmp-$SPARK_BENCH-$ts" /HiBench/bin/workloads/${S["$SPARK_BENCH"]}/$SPARK_BENCH/spark/run.sh >> ${OUTPUT_DIR}/$ts.log 2>&1
    # Move results to output folder
    docker exec "tmp-$SPARK_BENCH-$ts" mv /HiBench/report/hibench.report /results/report.log >> ${OUTPUT_DIR}/$ts.log 2>&1
	sleep 2s

	echo "Killing monitoring"
	kill -2 $pid
	echo "Killing containers"
	docker rm -f $(docker ps | awk '{print $NF}' | grep -v NAMES)
	sleep 10s
done; done; done