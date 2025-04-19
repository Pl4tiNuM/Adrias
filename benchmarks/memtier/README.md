# Memtier Benchmarking Tool
A High-Throughput Benchmarking Tool for Redis & Memcached

## Build
This image is based on the official `Dockerfile` found in https://github.com/RedisLabs/memtier_benchmark.git

To build the image:
```bash
docker build -t memtier .
```

## Run
This benchmark evaluates either a redis or a memcached database. Before you run this container you should first deploy either a [redis server]([../redis) or a [memcached database](../memcached)

Then simply run
```bash
docker run --rm \
--name=memtier-benchmark \
-v <localresults folder>:/results \
--net=<redis-network,memcached-network> \
<redis,memcached> \
<number of requests> \
<number of clients> \
<set:get ratio> \
<duration> \
<out_file>