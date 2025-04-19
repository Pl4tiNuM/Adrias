# REDIS in-memory database

Redis is an open source (BSD licensed), in-memory data structure store, used as a database, cache, and message broker.

## Build
This image builds upon the official `redis` image and just installs `numactl` to support redis deployment over disaggregated memory.

To build the image:
```bash
docker build -t redis .
```

## Run
To run the server we first have to create a local docker network
```
docker network create redis-network
```

Then just run the container
```
docker run --rm -d -P --privileged --net=redis-network --name=redis-server redis <local,remote>
```
`local` and `remote` define whether to run the server on local or disaggregated memory

Other containers should be able to connect to the server on the network through redis-server:6379

## Benchmark
Check `memtier` benchmark in parent directory