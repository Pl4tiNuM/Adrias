#!/bin/bash

CONFIG_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"

MONITOR_PATH="${CONFIG_PATH}/00-monitor"
PROFILER_PATH="${CONFIG_PATH}/01-profile"
SIMULATOR_PATH="${CONFIG_PATH}/02-simulate"
TRAINER_PATH="${CONFIG_PATH}/03-train"
PREDICTOR_PATH="${CONFIG_PATH}/04-predict"
ORCHESTRATOR_PATH="${CONFIG_PATH}/05-orchestrate"

CONTAINER_CPUS="0-63"
CONTAINER_MEMORY="16g"

rw_pipe() {
	val=$1
	(
		flock -s 200
		. .pipe
		val=$(( NUM_JOBS + $val ))
		echo "NUM_JOBS=$val" > .pipe
	) 200>/var/lock/pipe.lock
}

plog() {
	echo $1 | tee -a wrapper.log
}

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