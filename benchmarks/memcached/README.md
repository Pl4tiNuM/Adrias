# Memcached in-memory database

Memcached is a free & open source, high-performance, distributed memory object caching system

## Build
This image builds upon the official `memcached` image and just installs `numactl` to support redis deployment over disaggregated memory.

To build the image:
```bash
docker build -t memcached .
```

## Run
To run the server we first have to create a local docker network
```
docker network create memcached-network
```

Then just run the container
```
docker run --rm -d -P --privileged --net=memcached-network --name=memcached-server memcached <local,remote>
```
`local` and `remote` define whether to run the server on local or disaggregated memory

Other containers should be able to connect to the server on the network through memcached-server:11211

## Benchmark
Check `memtier` benchmark in parent directory