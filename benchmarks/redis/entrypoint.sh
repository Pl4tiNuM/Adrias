#!/bin/bash

mode=$1

if [ "$mode" = "local" ]; then
	numactl --membind 0 redis-server
elif [ "$mode" = "remote" ]; then
	numactl --membind 16 redis-server
fi
