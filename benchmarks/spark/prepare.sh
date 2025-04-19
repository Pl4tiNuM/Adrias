#!/bin/bash

${SPARK_HOME}/sbin/start-master.sh

sleep 5s

URL=$(cat /opt/spark/logs/* | grep "Starting Spark master at" | awk '{print $NF}')

${SPARK_HOME}/sbin/start-slave.sh $URL

sleep 5s

echo "hibench.spark.master $URL" >> /HiBench/conf/spark.conf
