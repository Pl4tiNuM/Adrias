#!/bin/bash

SCRIPT_PATH="$( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"

source ${SCRIPT_PATH}/../config.sh

interval=$1
duration=$2

_main() {
    mkdir -p "scenarios"
    OUT_DIR="scenarios/$interval-$duration-$s"
    t_total=0
    while :; do
        # Choose benchmark randomly
        rand=$[$RANDOM % ${#B[@]}]
        bench=${B[$rand]}

        # Random arrival time of next job (between 5-60 seconds)
        t_next=$(shuf -i "$interval" -n 1)
        t_total=$(( t_total+t_next ))
        echo "${bench},${t_next}s" >> ${OUT_DIR}
        if [ "$t_total" -gt "$duration" ]; then
            break
        fi
    done
}

_main
