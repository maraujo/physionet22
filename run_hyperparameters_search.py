# curl http://algodev.matheusaraujo.com:8032/ohh.config --output ohh.config

import os
import glob
import uuid
import loguru

loguru.info("Generate new ohh.config")

def generate_new_ohh_config_given():
    pass
os.system("curl http://algodev.matheusaraujo.com:8032/parameters.csv --output parameters.csv")

github_token = input("Type github token: ")

os.system("apt install -y vim")
os.system("apt install -y htop")
os.system("apt install -y libsndfile1")
os.system("apt install -y unzip")
os.system("rm -r /physionet_data/challenge/files/cross-validation-data-1-0-3/")

os.system("mkdir -p /physionet_data/challenge/files/cross-validation-data-1-0-3/")
os.system("mkdir -p /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.system("git clone --branch matheus https://matheus:{}@github.com/maraujo/physionet22.git".format(github_token))
os.system("pip install -r ./physionet22/requirements.txt")
os.system("pip install tensorflow==2.8.2")

import boto3

os.system("rm the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("wget https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("unzip -q -o the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") 
os.system("mv ./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.system("cp ohh.config physionet22/")
os.chdir('physionet22/')
os.system("python generate_crossvalidation_splits.py")
os.system("python ./test_code_crossvalidation_splits.py")
