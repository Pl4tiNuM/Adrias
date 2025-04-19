# Perf events to monitor
# CONTAINER_PERF_EVENTS = [
#      'cache-misses',
#      'cache-references',
#      'cpu-cycles',
#      'L1-dcache-load-misses',
#      'L1-dcache-loads',
#      'L1-dcache-prefetches',
#      'L1-dcache-store-misses',
#      'L1-icache-load-misses',
#      'L1-icache-loads',
#      'L1-icache-prefetches',
#      'L2-load-misses'
# ]

# SYSTEM_PERF_EVENTS = [
#      'LLC-load-misses',
#      'LLC-loads',
#      'LLC-prefetches',
#      'mem-loads',
#      'mem-stores'
# ]

CONTAINER_PERF_EVENTS = [
   'LLC-load-misses',
   'LLC-loads'
]

SYSTEM_PERF_EVENTS = [
     'LLC-load-misses',
     'LLC-loads',
     'mem-loads',
     'mem-stores'
]

 # 1 second monitoring interval
CONTAINER_MONITORING_INTERVAL = 1000
SYSTEM_MONITORING_INTERVAL = 1000

THYMESISFLOW_HOME="/home/pl4tinum/Desktop/"
