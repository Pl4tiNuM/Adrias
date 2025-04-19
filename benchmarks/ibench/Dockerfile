FROM gcc

RUN apt-get update && apt-get -y install numactl vim

ADD . /ibench

RUN cd /ibench && make

ENV PATH="/ibench/src:${PATH}"

WORKDIR /ibench/src

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
