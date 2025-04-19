#!/bin/bash

/etc/init.d/ssh start

mode=$1

${HADOOP_HOME}/bin/hdfs namenode -format
${HADOOP_HOME}/sbin/start-dfs.sh

${SPARK_HOME}/sbin/start-master.sh

sleep 1s
while [ -z "${MASTER_URL}" ]; do
	echo "waiting for Spark master to initiate..."
	MASTER_URL=$(cat /opt/spark/logs/spark--org.apache.spark.deploy.master.Master-1-*.out | grep spark: | awk '{print $NF}')
	sleep 1s
done

# If local mode start spark slave on local
if [ "$mode" = "local" ]; then
	numactl --membind 0 ${SPARK_HOME}/sbin/start-slave.sh ${MASTER_URL}
# else remote
elif [ "$mode" = "remote" ]; then
	numactl --membind 16 ${SPARK_HOME}/sbin/start-slave.sh ${MASTER_URL}
fi

while [ -z "${WORKER_UP}" ]; do
	echo "waiting for Spark worker to initiate..."
	WORKER_UP=$(cat /opt/spark/logs/spark--org.apache.spark.deploy.worker.Worker-1-*.out | grep "Successfully registered")
	sleep 1s
done

echo "hibench.spark.master ${MASTER_URL}" >> ${HIBENCH_CONF_DIR}/spark.conf

while :; do
	echo "Running..."
	sleep 10s
done
