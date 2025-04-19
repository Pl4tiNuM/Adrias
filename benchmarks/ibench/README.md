# iBench
This is a clone of the official iBench repository: https://github.com/stanford-mast/iBench

## Modifications
The Dockerfile has been modified to support the memory disaggregation, i.e., installed `numactl` and provided a simple `entrypoint`

## Building the image
Just run
```bash
docker build -t ibench .
```

## Running the benchmarks
Just run
```bash
docker run --rm --privileged ibench <memory_mode> <stressing_resource>
```

where `memory mode` is one of `["local","remote"]` and `stressing_resource` one of `["l2","l3","cpu","memBw","memCap"]`

Privileged is required to enable the NUMA policies.