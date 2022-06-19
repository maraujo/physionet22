#FROM python:3.7.13-buster
FROM tensorflow/tensorflow:latest-gpu

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER limadim@ccf.org


RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 ffmpeg 

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
