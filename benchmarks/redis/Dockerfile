FROM redis
RUN apt update && apt-get update
RUN apt-get -y install numactl lsof iputils-ping

WORKDIR /
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh

EXPOSE 6379

ENTRYPOINT ["./entrypoint.sh"]
