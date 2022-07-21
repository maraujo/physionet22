import os
os.system("pip install loguru==0.5.3")
os.system("pip install pandas==1.3.5")
os.system("pip install psycopg2-binary==2.9.1")
os.system("pip install sqlalchemy-cockroachdb==1.4.3")
import glob
import uuid
from loguru import logger
import os
from sqlalchemy import create_engine, text, update
import pandas as pd
import sys
import logging
import json
import pprint
import urllib.request
import urllib.parse

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
PATIENCE_DAYS = 2
github_token = "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"




os.system("curl --create-dirs -o $HOME/.postgresql/root.crt -O https://cockroachlabs.cloud/clusters/6cadd36b-9892-418c-88c7-64a5781755ec/cert")

dir_path = os.getcwd()

try:
    os.system("apt install -y vim")
    os.system("apt install -y htop")
    os.system("apt install -y libsndfile1")
    os.system("apt install -y unzip")
except:
    logger.warning("You need these libs installed.")






#Cleaning from previous run    
os.system("rm -r ./cross-validation-data-1-0-3/")
os.system("rm -r ./circor-heart-sound/")
os.system("rm ./murmur_final_result_current.csv")
os.system("rm ./outcome_final_result_current.csv")
os.system("rm -r ./circor-heart-sound/")

os.system("mkdir -p ./cross-validation-data-1-0-3/")
os.system("mkdir -p ./circor-heart-sound/1.0.3/")
os.system("git clone --branch projectx https://matheus:{}@github.com/maraujo/physionet22.git".format(github_token))
assert os.system("pip install -r ./physionet22/requirements.txt") == 0
assert os.system("pip install tensorflow==2.8.2") == 0

os.system("rm the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
assert os.system("wget https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") == 0
assert os.system("unzip -q -o the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") == 0
assert os.system("mv ./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data ./circor-heart-sound/1.0.3/") == 0

os.chdir('physionet22/')
os.system("python generate_crossvalidation_splits.py")
assert os.system("python ./test_code_crossvalidation_splits.py") == 1

murmur_df = pd.read_csv("../murmur_final_result_current.csv")
outcome_df = pd.read_csv("../outcome_final_result_current.csv")
murmur_mean = murmur_df['Weighted Accuracy'].mean()
murmur_std = murmur_df['Weighted Accuracy'].std()
outcome_mean = outcome_df['Cost'].mean()
outcome_std = outcome_df['Cost'].std()

text_result = ""
text_result += urllib.parse.quote(open("ohh.config").read() + "\n")
text_result += urllib.parse.quote(open("../murmur_final_result_current.csv").read() + "\n")
text_result += urllib.parse.quote(open("../outcome_final_result_current.csv").read() + "\n")
urllib.request.urlopen("https://vorkqcranza3s6f66wloniatvy0duufg.lambda-url.us-east-1.on.aws/?destiny=matheus.ld.araujo@gmail.com&text={}&password=FIYl4lXi6QHMJHth&subject=DoneLambda".format(text_result))
os.chdir(dir_path)