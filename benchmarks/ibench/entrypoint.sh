#!/bin/bash

mode=$1

rsrc=$2

if [ "$mode" = "local" ]; then
	numactl --membind 0,8 ./$rsrc 1000000
elif [ "$mode" = "remote" ]; then
	numactl --membind 16 ./$rsrc 1000000
fi
