FROM python:3.7

RUN apt update && apt-get update
RUN apt-get -y install gfortran libopenblas-dev liblapack-dev

RUN pip3 install numpy scipy sklearn
RUN pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

