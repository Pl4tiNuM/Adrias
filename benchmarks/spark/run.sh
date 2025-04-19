#!/bin/bash

BENCH=$1
BENCH_PATH=$2
DATASET=$3

# Change the dataset size
sed -i "s/hibench.scale.profile.*/hibench.scale.profile $DATASET/" /HiBench/conf/hibench.conf
# Create the dataset
${BENCH_PATH}/prepare/prepare.sh
# Execute for all the configurations
for CONFIG in $(ls /configs)
do
	# Run the benchmark
        export MY_SPARK_CONF=/configs/$CONFIG
                ${BENCH_PATH}/spark/run.sh

                mkdir -p /results/$BENCH/${DATASET}/$CONFIG/
                cp /HiBench/report/$BENCH/spark/bench.log /results/$BENCH/$DATASET/$CONFIG/bench.log
                mv /HiBench/report/hibench.report /results/$BENCH/$DATASET/$CONFIG/report.log
                sleep 30s
        done

