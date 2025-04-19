FROM memcached
USER root
RUN apt-get update
RUN apt-get -y install numactl

WORKDIR /
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 11211
ENTRYPOINT ["./entrypoint.sh"]

