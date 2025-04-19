#!/bin/bash


set -x 
target=$1
requests=$2
clients=$3
ratio=$4
output=$5
server=$6


if [ "$target" = "redis" ]; then
	memtier_benchmark --hide-histogram \
		--server=$server \
		--port=6379 \
		--requests=$requests \
		--clients=$clients \
		--ratio=$ratio \
		--out-file=/results/$output
elif [ "$target" = "memcached" ]; then
        memtier_benchmark --hide-histogram \
                --server=$server \
                --port=11211 \
		--protocol=memcache_text \
                --requests=$requests \
                --clients=$clients \
                --ratio=$ratio \
		--out-file=/results/$output
fi
