#!/bin/bash

SCRIPT_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"

stress_jobs=("l2" "l3" "cpu" "memBw" "memCap")

function _init() {

	# Check if script is ran as root
	if [ "$EUID" -ne 0 ]
	then
		echo "Please run as root"
		exit
	fi

	# Number of cores
	CORES=63

	# Get current timestamp
	ts=$(date +"%s")

	# Create output folder
	mkdir -p ${SCRIPT_PATH}/output
}

function _main() {

	for (( x = 0; x < 100; x++ ))
	do
		# Eliminating CPU and JOB array
		for i in {0..63}
		do
			PID[$i]=0
			JOB[$i]=0
		done
		
		# Get current timestamp
		ts=$(date +"%s")
		# Create results folder
		mkdir -p interference-profiling/${ts}
		# Create log file
		touch interference-profiling/${ts}/${ts}.log

		# Firstly, let's start our stressing jobs
		# randomly selected between 2 and 60.
		no_stress_jobs=$(shuf -i 2-60 -n 1)
		printf "Stress jobs to run : %s\n" "${no_stress_jobs}" | tee -a interference-profiling/${ts}/${ts}.log
		printf "===========================================\n" | tee -a interference-profiling/${ts}/${ts}.log
		printf "AA\tCORE\tWORKLOAD\tPORTION\tPID\n" | tee -a interference-profiling/${ts}/${ts}.log

		# Spawn stressing jobs
		for (( i = 0; i < ${no_stress_jobs}; i++ ))
		do
			# Select the CPU to run the job @
			id=$(shuf -i 1-$CORES -n 1)
			# Ensure that no other job is running on this core
			while (( PID[$id] != 0 ))
			do
				id=$(shuf -i 0-$CORES -n 1)
			done
			# Flag
			PID[$id]=1
			# Which resource should i stress?
			index=$(shuf -i 0-6 -n 1)
			resource=${stress_jobs[${index}]}
			JOB[$id]=$resource

			docker run --rm --cpus=1 --cpuset-cpus=0-63 --entrypoint "./$resource" pl4tinum/ibench "1000000" &
			PID[$id]=$!
			fi
			printf "$i\t$id\t$resource\t${PID[$id]}\n"  | tee -a interference-profiling/${ts}/${ts}.log
			sleep 0.1s
		done
		printf "===========================================\n" | tee -a interference-profiling/${ts}/${ts}.log
		# Now that the stressing scenarios are running
		# i start monitoring my performance counters
		echo -n "Starting monitoring performance counters... " | tee -a interference-profiling/${ts}/${ts}.log
	    	./pcm.x $delay -r -csv=pcm.csv 1>&- 2>&- &
		echo "Success" | tee -a interference-profiling/${ts}/${ts}.log

		# Now let us also start our real application!
		id=$(shuf -i 0-47 -n 1)
		while (( PID[$id] != 0 ))
		do
			id=$(shuf -i 0-$CORES -n 1)
		done
		echo "Application to run on core $id" | tee -a interference-profiling/${ts}/${ts}.log
		#docker run --rm \
		#--cpus=1 \
		#--cpuset-cpus=$id \
		#--volumes-from data cloudsuite/graph-analytics \
		#--driver-memory 8g \
		#--executor-memory 8g

		#docker run --rm \
		#--cpus=1 \
		#--cpuset-cpus=$id \
		#-it \
		#--name client \
		#--net search_network \
		#cloudsuite/web-search:client 172.20.0.2 100 90 60 60

		# docker run --rm \
		# --cpus=1 \
		# --cpuset-cpus=$id \
		# --volumes-from inmemdata \
		# cloudsuite/in-memory-analytics \
		# /data/ml-latest \
		# /data/myratings.csv \
		# --driver-memory 8g \
		# --executor-memory 8g

		# Run our application
		$params --cpus=1 --cpuset-cpus=$id "$image"

		# Kill PCM
		echo -n "Killing pcm script... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "pcm.x"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill l1d stress job
		echo -n "Killing l1d... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "l1d"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill l1i stress job
		echo -n "Killing l1i... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "l1i"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill l2 stress job
		echo -n "Killing l2... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "l2"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill l3 stress job
		echo -n "Killing l3... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "l3"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill cpu stress job
		echo -n "Killing cpu... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "cpu"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill memCap stress job
		echo -n "Killing memCap... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "memCap"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log

		# Kill memBw stress job
		echo -n "Killing memBw... " | tee -a interference-profiling/${ts}/${ts}.log
		killall -9 "memBw"
		echo "Done" | tee -a interference-profiling/${ts}/${ts}.log


		# TODO copy results to folder
		echo "Moving output files"
		echo "pcm.csv -> interference-profiling/${ts}/pcm.csv"
		mv pcm.csv interference-profiling/${ts}/pcm.csv
		echo "Done"
		sleep 2s
	done
}

_init
_main