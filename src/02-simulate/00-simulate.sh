#!/bin/bash

SCRIPT_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"
OUTPUT_DIR=$(pwd)/"results/"
MONITOR_HOME="/nfs_homes/dmasouros/project/src/monitor"

B=( "l2" "l3" "cpu" "memBw" "memCap" \
"redis" "memcached" \
"nweight" "repartition" "sort" "terasort" "wordcount" \
"als" "bayes" "gbt" "gmm" "kmeans" "lda" "lr" "pca" "rf" "svd" "svm" "pagerank")

declare -A S
S=( \
	["nweight"]="graph" ["sort"]="micro" ["terasort"]="micro" ["wordcount"]="micro" ["repartition"]="micro" \
	["rf"]="ml" ["lr"]="ml" ["als"]="ml" ["gbt"]="ml" ["gmm"]="ml" ["gbt"]="ml" ["lda"]="ml" ["pca"]="ml" ["svd"]="ml" ["svm"]="ml" ["bayes"]="ml" ["kmeans"]="ml" \
	["pagerank"]="websearch" \
)

interval=$1
duration=$2

start_ibench(){
    local mode=$1
    local duration=$2
    local rsrc=$3
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/$rsrc/$sts
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
    plog "[$sts] start,$rsrc,ibench,$mode,$dur"
    echo "[$sts] start,$rsrc,ibench,$mode,$dur" >> ${OUT_DIR}/$sts.log
    docker run -d --rm --privileged --cpuset-cpus=0-63 --memory=16g \
                --name "$rsrc-$sts" \
                dmasouros/ibench $mode $rsrc &
    sleep $duration
    docker rm -f $rsrc-$sts
	rw_pipe "-1"
    local ets=$(date +"%s")
    plog "[$ets] end,$rsrc,ibench,$mode,$dur"
    echo "[$ets] end,$rsrc,ibench,$mode,$dur" >> ${OUT_DIR}/$sts.log
    return
}

start_redis(){
    local mode=$1
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/redis/$sts
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
    plog "[$sts] start,redis,redis,$mode"
    echo "[$sts] start,redis,redis,$mode" >> ${OUT_DIR}/$sts.log
    # Create docker network
    docker network create "redis-network-$sts" >> ${OUT_DIR}/$sts.log 2>&1
    # Run server either local or remote
    docker run -d -P --rm --privileged --cpuset-cpus=0-63 \
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
    local mode=$1
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/memcached/$sts
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
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
    local mode=$1
    local lbench=$2
    local sts=$(date +"%s")
    local OUT_DIR=${OUTPUT_DIR}/$lbench/$sts
    local SYSTEM_UP=""
    mkdir -p ${OUT_DIR}
	rw_pipe "1"
    plog "[$sts] start,$lbench,${S["$lbench"]},spark,$mode"
    echo "[$sts] start,$lbench,${S["$lbench"]},spark,$mode" >> ${OUT_DIR}/$sts.log 2>&1 
    # Run spark either local or remote. We first name the container as tmp, so that the watcher does not monitor it
    docker run -d -P --rm --privileged --cpuset-cpus=0-63 \
               -v ${OUT_DIR}:/results \
               --name "tmp-$lbench-$sts" \
               dmasouros/spark $mode >> ${OUT_DIR}/$sts.log 2>&1
    # Wait for server to come up
    while [ -z "${SYSTEM_UP}" ]; do
	    echo "Waiting for system to come up"
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

rw_pipe() {
	val=$1
	(
		flock -s 200
		. .pipe
		val=$(( NUM_JOBS + $val ))
		echo "NUM_JOBS=$val" > .pipe
	) 200>/var/lock/pipe.lock
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
	rm -r wrapper.log
	rm -r .pipe
	rm -r ${OUTPUT_DIR}
	mkdir -p ${OUTPUT_DIR}
	# nproc gives scale (1-cores) - convert to 0-(cores-1)
	CORES=$(( CORES-1 ))
}

plog() {
	echo $1 | tee -a wrapper.log
}

_main() {

	plog "--- Starting monitoring in background"
	# Start monitoring
	python3 -u ${MONITOR_HOME}/watcher.py -c 1000 -s 1000 -o "${OUTPUT_DIR}/" >> $OUTPUT_DIR/watcher.log 2>&1 &
	pid=$!
	while :; do top -b -d 1 -p $pid >> $OUTPUT_DIR/utilization.log 2>&1; done &
	top_pid=$!
    plog "--- Starting simulation"
	echo "NUM_JOBS=0" > .pipe
    touch wrapper.log
	start_time=$(date +%s)
	while :; do

        # Choose benchmark randomly
        rand=$[$RANDOM % ${#B[@]}]
        bench=${B[$rand]}

        # Choose mode randomly
        mode=$(shuf -n1 -e "local" "remote")

        # Run benchmark
        if [ "$bench" = "cpu" ] || [ "$bench" = "l2" ] || [ "$bench" = "l3" ] || [ "$bench" = "memBw" ] || [ "$bench" = "memCap" ]; then
            dur=$(shuf -i 10-90 -n 1)
            start_ibench "$mode" "$dur" "$bench" &
        elif [ "$bench" = "redis" ]; then
            start_redis "$mode" &
        elif [ "$bench" = "memcached" ]; then
            start_memcached "$mode" &
        else
            start_spark "$mode" "$bench" &
        fi

        # Random arrival time of next job (between 5-60 seconds)
        t_next=$(shuf -i "$interval" -n 1)
		. .pipe && plog "[$(date +"%s")] Current active jobs: ${NUM_JOBS}"
        plog "[$(date +"%s")] Next job in ${t_next}s"
        sleep ${t_next}
		current_time=$(date +%s)
		elapsed=$(( current_time - start_time ))
		plog "[$current_time] Elapsed time: ${elapsed}s"
		if [ "$elapsed" -gt "$duration" ]; then
			plog "[$current_time] Simulation period exceeded"
			plog "[$current_time] Waiting for current running workloads to finish"
			while [ "$NUM_JOBS" -ne 0 ]; do
				sleep 1s
				plog "[$(date +%s)] Current active jobs: ${NUM_JOBS}"
				. .pipe
			done
			break
		fi
	done
	plog "--- Finalizing simulation"
	plog "--- Killing monitoring"
	kill -9 $top_pid
	kill -2 $pid
	sleep 15s
	mkdir "$interval-$duration"
	mv results "$interval-$duration"
	mv wrapper.log "$interval-$duration"
}

_init
_main
