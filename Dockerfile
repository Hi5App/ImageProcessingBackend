# Use an official Python runtime as a parent image
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container to /app
WORKDIR /app

ADD . /app

RUN apt update -y

RUN apt upgrade -y

RUN apt install git -y

RUN apt install software-properties-common -y

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update -y

RUN apt install python3.9 -y 

RUN apt install pip -y

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

CMD ["python", "InferenceServer.py"]
