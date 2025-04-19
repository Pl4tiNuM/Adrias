# Spark in-memory computing

Apache Spark is an open-source, distributed processing system used for big data workloads. It utilizes in-memory caching and optimized query execution for fast queries against data of any size.

## Build
This image builds from scratch from `ubuntu` image and installs Hadoop HDFS in a pseudo-distributed mode as well as `numactl` to support spark deployment over disaggregated memory.
It also includes HiBench benchmark suite, for evaluating the remote memory system.

To build the image:
```bash
docker build -t spark .
```

## Run
To execute the benchmarks we first run the container
```bash
docker run -d -P --rm --privileged --name spark dmasouros/spark <local,remote>
```
`local` and `remote` define whether to run the Spark Worker on local or disaggregated memory.

After the Worker has been initiated (can check with `docker logs`), we can execute an application from HiBench suite.

First  create the respective dataset:
```bash
docker exec "spark" /HiBench/bin/workloads/<benchmark-family>/<benchmark>/prepare/prepare.sh
```
where `<benchmark-family>` and `<benchmark>` is the class (e.g., micro, ml, etc.) and the application (e.g., terasort, bayes) respectively

Finally, we can run the benchmark
```
docker exec "spark" /HiBench/bin/workloads/<benchmark-family>/<benchmark>/spark/run.sh
```