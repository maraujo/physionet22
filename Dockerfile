FROM python:3.8.12-buster

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER limadim@ccf.org


RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -U pip
RUN pip install -r requirements.txt
