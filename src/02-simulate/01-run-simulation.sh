#!/bin/bash

SCRIPT_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"
source ${SCRIPT_PATH}/../config.sh

OUTPUT_DIR=$(pwd)/"results/"

SCENARIO=$1
SCHEDULER=$2

start_ibench(){
    local duration=$1
    local rsrc=$2
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/$rsrc/$sts
    local CONTAINER_NAME="$rsrc-$sts"
    mkdir -p ${OUT_DIR}
	rw_pipe "1"

    mode=$(python3 ${ORCHESTRATOR_PATH}/spawner.py -b "$rsrc" -f "ibench")

    plog "[$sts] start,$rsrc,ibench,$mode,$dur"
    echo "[$sts] start,$rsrc,ibench,$mode,$dur" >> ${OUT_DIR}/$sts.log
    docker run -d --rm  --privileged 
                        --cpuset-cpus=${CONTAINER_CPUS} \
                        --memory=${CONTAINER_MEMORY} \
                        --name=${CONTAINER_NAME} \
                    dmasouros/ibench $rsrc $mode &
    sleep $duration
    docker rm -f $rsrc-$sts
	rw_pipe "-1"
    local ets=$(date +"%s")
    plog "[$ets] end,$rsrc,ibench,$mode,$dur"
    echo "[$ets] end,$rsrc,ibench,$mode,$dur" >> ${OUT_DIR}/$sts.log
    return
}

start_redis(){
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/redis/$sts
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
    
    mode=$(python3 ${ORCHESTRATOR_PATH}/spawner.py -b "redis" -f "redis")
    plog "[$sts] start,redis,redis,$mode"
    echo "[$sts] start,redis,redis,$mode" >> ${OUT_DIR}/$sts.log
    # Create docker network
    docker network create "redis-network-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    # Run server either local or remote
    docker run -d -P --rm --privileged 
                        --cpuset-cpus=${CONTAINER_CPUS} \
                        --name "redis-server-$sts" \
               --network "redis-network-$sts" dmasouros/redis $mode >> ${OUT_DIR}/$sts.log 2>&1
    # Wait for server to come up
    sleep 3s
    # Run benchmark
    docker run --rm --cpuset-cpus=64-127 \
	           -v "${OUT_DIR}":/results \
               --name="memtier-benchmark-$sts" \
               --net="redis-network-$sts" \
		       dmasouros/memtier \
                    "redis" \
                    "10000" \
                    "200" \
                    "1:10" \
                    "$sts.log" \
		    "redis-server-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    docker rm -f "redis-server-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    docker network rm "redis-network-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    rw_pipe "-1"
    local ets=$(date +"%s")
    plog "[$ets] end,redis,redis,$mode"
    echo "[$ets] end,redis,redis,$mode" >> ${OUT_DIR}/$sts.log 2>&1
    return
}

start_memcached(){
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/memcached/$sts
    mkdir -p ${OUT_DIR}
	rw_pipe "1"

    mode=$(python3 ${ORCHESTRATOR_PATH}/spawner.py -b "memcached" -f "memcached")
    plog "[$sts] start,memcached,memcached,$mode"
    echo "[$sts] start,memcached,memcached,$mode" >> ${OUT_DIR}/$sts.log 2>&1

    # Create docker network
    docker network create "memcached-network-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    # Run server either local or remote
    docker run -d -P --rm --privileged --cpuset-cpus=0-63 \
               --name "memcached-server-$sts" \
               --network "memcached-network-$sts" dmasouros/memcached $mode >> ${OUT_DIR}/$sts.log 2>&1
    # Wait for server to come up
    sleep 3s
    # Run benchmark
    docker run --rm --cpuset-cpus=64-127 \
	           -v "${OUT_DIR}":/results \
               --name="memtier-benchmark-$sts" \
               --net="memcached-network-$sts" \
		       dmasouros/memtier \
                    "memcached" \
                    "40000" \
                    "200" \
                    "1:10" \
                    "$sts.log" \
		    "memcached-server-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    docker rm -f "memcached-server-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    docker network rm "memcached-network-$sts" >> ${OUT_DIR}/$sts.log 2>&1
	rw_pipe "-1"
    local ets=$(date +"%s")
    plog "[$ets] end,memcached,memcached,$mode"
    echo "[$ets] end,memcached,memcached,$mode" >> ${OUT_DIR}/$sts.log 2>&1
    return
}

start_spark(){
    local lbench=$1
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/$lbench/$sts
    local SYSTEM_UP=""
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
    mode=$(python3 ${ORCHESTRATOR_PATH}/spawner.py -b "$lbench" -f "spark")
    plog "[$sts] start,$lbench,${S["$lbench"]},spark,$mode"
    echo "[$sts] start,$lbench,${S["$lbench"]},spark,$mode" >> ${OUT_DIR}/$sts.log 2>&1 
    # Run spark either local or remote. We first name the container as tmp, so that the watcher does not monitor it
    docker run -d -P --rm --privileged --cpuset-cpus=0-63 \
               -v ${OUT_DIR}:/results \
               --name "tmp-$lbench-$sts" \
               dmasouros/spark $mode >> ${OUT_DIR}/$sts.log 2>&1
    # Wait for server to come up
    while [ -z "${SYSTEM_UP}" ]; do
	    SYSTEM_UP=$(docker logs "tmp-$lbench-$sts" 2>&1 | grep "Running...")
	    sleep 1s
    done
    # Change dataset size to small and create it
    docker exec "tmp-$lbench-$sts" sed -i 's/tiny/small/g' /HiBench/conf/hibench.conf >> ${OUT_DIR}/$sts.log 2>&1
    docker exec "tmp-$lbench-$sts" /HiBench/bin/workloads/${S["$lbench"]}/$lbench/prepare/prepare.sh >> ${OUT_DIR}/$sts.log 2>&1

    # Rename container to start monitoring
    docker rename "tmp-$lbench-$sts" "spark-$bench-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    sleep 2s
    # Run benchmark
    docker exec "spark-$lbench-$sts" /HiBench/bin/workloads/${S["$lbench"]}/$lbench/spark/run.sh >> ${OUT_DIR}/$sts.log 2>&1
    # Move results to output folder
    docker exec "spark-$lbench-$sts" mv /HiBench/report/hibench.report /results/report.log >> ${OUT_DIR}/$sts.log 2>&1

    docker rm -f "spark-$lbench-$sts" >> ${OUT_DIR}/$sts.log 2>&1
	rw_pipe "-1"
    local ets=$(date +"%s")
    plog "[$ets] end,$lbench,spark,$mode"
    echo "[$ets] end,$lbench,spark,$mode" >> ${OUT_DIR}/$sts.log 2>&1
    return
}

_init() {

	# Check if script is ran as root
	if [ "$EUID" -ne 0 ]
	then
		echo "Please run as root"
		exit
	fi

	# Number of cores
	CORES=$(nproc)
	printf "Number of cores : $CORES\n\n"
	rm -rf wrapper.log
	rm -rf .pipe
	rm -rf ${OUTPUT_DIR}
	mkdir -p ${OUTPUT_DIR}
	# nproc gives scale (1-cores) - convert to 0-(cores-1)
	CORES=$(( CORES-1 ))
}

_main() {
    plog "Starting simulation for ${SCENARIO}"
	plog "--- Starting monitoring in background"
	# Start monitoring
	python3 -u ${MONITOR_PATH}/watcher.py -c 1000 -s 1000 -o "${OUTPUT_DIR}/" -d "${SCHEDULER}" >> $OUTPUT_DIR/watcher.log 2>&1 &
	pid=$!
    plog "--- Starting simulation"
	echo "NUM_JOBS=0" > .pipe
    touch wrapper.log
	while IFS= read -r line; do
        bench=$(echo $line | awk -F, '{print $1}')
        t_next=$(echo $line | awk -F, '{print $2}')
        # Run benchmark
        if [ "$bench" = "cpu" ] || [ "$bench" = "l2" ] || [ "$bench" = "l3" ] || [ "$bench" = "memBw" ] || [ "$bench" = "memCap" ]; then
            dur=$(shuf -i 10-90 -n 1)
            start_ibench "$dur" "$bench" &
        elif [ "$bench" = "redis" ]; then
            start_redis &
        elif [ "$bench" = "memcached" ]; then
            start_memcached &
        else
            start_spark "$bench" &
        fi
        sleep 0.1

		. .pipe && plog "[$(date +"%s")] Current active jobs: ${NUM_JOBS}"
        sleep ${t_next}

	done < "$SCENARIO"

    plog "[$current_time] All simulation jobs submitted"
    plog "[$current_time] Waiting for current running workloads to finish"
    while [ "$NUM_JOBS" -ne 0 ]; do
        sleep 1s
        plog "[$(date +%s)] Current active jobs: ${NUM_JOBS}"
        . .pipe
    done
	plog "--- Finalizing simulation"
	plog "--- Killing monitoring"
	kill -2 $pid
	sleep 15s
	mkdir "$interval-$duration"
	mv results "$interval-$duration"
	mv wrapper.log "$interval-$duration"
}

_init
_main
