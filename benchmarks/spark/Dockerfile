FROM ubuntu

# Ignore interactions during installation of packages
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get -y update && apt-get -y -qq install \
	maven \
	scala \
	vim \
	wget \
	python2.7 \
	software-properties-common \
	git \
	bc \
	numactl \
	ssh
RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - && \
    add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ && \
    apt update && \
    apt -y install adoptopenjdk-8-hotspot

WORKDIR /usr/bin
RUN ln -s python2.7 python2 && ln -s python2.7 python

# Install Hadoop
WORKDIR /usr/local
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-2.7.4/hadoop-2.7.4.tar.gz && \
    tar xzvf hadoop-2.7.4.tar.gz && \
    rm hadoop-2.7.4.tar.gz && \
    mv hadoop-2.7.4 hadoop
ENV HADOOP_HOME=/usr/local/hadoop
ENV HADOOP_INSTALL=/usr/local/hadoop
ENV HADOOP_PREFIX /usr/local/hadoop
ENV HADOOP_COMMON_HOME /usr/local/hadoop
ENV HADOOP_HDFS_HOME /usr/local/hadoop
ENV HADOOP_MAPRED_HOME /usr/local/hadoop
ENV HADOOP_YARN_HOME /usr/local/hadoop
ENV HADOOP_CONF_DIR /usr/local/hadoop/etc/hadoop
ENV HADOOP_EXAMPLES_JAR=/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.4.jar
COPY conf/hdfs/core-site.xml ${HADOOP_CONF_DIR}/core-site.xml
COPY conf/hdfs/hdfs-site.xml ${HADOOP_CONF_DIR}/hdfs-site.xml
RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 0600 /root/.ssh/authorized_keys
COPY conf/ssh/ssh-config /root/.ssh/config

# Install spark
WORKDIR /opt
ENV JAVA_HOME=/usr/lib/jvm/adoptopenjdk-8-hotspot-ppc64el/jre
RUN echo "JAVA_HOME=${JAVA_HOME}" >> /etc/environment
RUN wget https://dlcdn.apache.org/spark/spark-3.0.3/spark-3.0.3.tgz
RUN tar xvf spark-3.0.3.tgz && \
    rm spark-3.0.3.tgz && \
    mv spark-3.0.3 spark

WORKDIR spark
RUN ./build/mvn -Pyarn -Phadoop-2.7 -Dscala-2.12 -DskipTests clean package
ENV SPARK_HOME=/opt/spark

# Install HiBench
WORKDIR /
COPY HiBench /HiBench
WORKDIR /HiBench
RUN mvn -Psparkbench -Dspark=3.0 -Dhadoop=2.7 -Dscala=2.12 clean package
ENV HIBENCH_HOME=/HiBench
ENV HIBENCH_CONF_DIR=/HiBench/conf/
COPY conf/hibench/spark.conf ${HIBENCH_CONF_DIR}/spark.conf
COPY conf/hibench/hadoop.conf ${HIBENCH_CONF_DIR}/hadoop.conf
WORKDIR /

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
