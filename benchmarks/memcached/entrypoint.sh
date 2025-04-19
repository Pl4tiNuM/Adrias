#!/bin/bash

mode=$1

if [ "$mode" = "local" ]; then
	numactl --membind 0 memcached -u nobody -m 4096 -c 2048 -M
elif [ "$mode" = "remote" ]; then
	numactl --membind 16 memcached -u nobody -m 4096 -c 2048 -M
fi