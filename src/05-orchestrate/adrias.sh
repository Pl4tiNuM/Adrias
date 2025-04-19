#!/bin/bash

SCRIPT_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"
source ${SCRIPT_PATH}/../config.sh

args="$@"

for (( i=1; i<=$#; i++)); do
    arg=${!i}
    # Next from --name is container name. Grab it
    if [ "$arg" == "--name" ]; then
        container_name="$((i+1))"
        container_name=${!container_name}
    fi
done

if [[ $container_name == *"redis"* ]]; then
    echo "It's redis"
    m=$(python3 ${SCRIPT_PATH}/spawner.py -b "redis" -f "redis")
fi

if [[ $container_name == *"memcached"* ]]; then
    echo "It's memcached"
    m=$(python3 ${SCRIPT_PATH}/spawner.py -b "memcached" -f "memcached")
fi

if [[ $container_name == *"tmp"* ]]; then
    bench=$(echo "$container_name" | awk -F'-' '{print $2}')
    m=$(python3 ${SCRIPT_PATH}/spawner.py -b "$bench" -f "spark")
fi

if [[ $container_name == *"ibench"* ]]; then
    bench=$(echo "$container_name" | awk -F'-' '{print $2}')
    m=$(python3 ${SCRIPT_PATH}/spawner.py -b "$bench" -f "ibench")
fi

# echo $m
docker $(echo ${@: 1:$#-1}) $m $(echo ${@: -1})