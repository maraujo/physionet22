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

def update_timestamp(timestamp, parameter_id):
    with engine.connect() as conn:
        if timestamp == None:
            sql_update_timestamp = "UPDATE parameters_search SET start_running_timestamp = NULL WHERE parameter_id = '{}';".format(parameter_id)
        else:
            sql_update_timestamp = "UPDATE parameters_search SET start_running_timestamp = '{}' WHERE parameter_id = '{}';".format(timestamp, parameter_id)
        res = conn.execute(text(sql_update_timestamp))
        logger.info("Updating timestamp: {} - {}".format(res, sql_update_timestamp))


def update_results(murmur_file, outcome_file, mean_murmur, std_murmur, std_outcome, mean_outcome, parameter_id):
    with engine.connect() as conn:
        sql_update_timestamp = "UPDATE parameters_search SET murmur_file = '{}', outcome_file = '{}', mean_murmur = '{}', std_murmur='{}', std_outcome='{}', mean_outcome='{}'  WHERE parameter_id = '{}';".format(murmur_file, outcome_file, mean_murmur, std_murmur, std_outcome, mean_outcome, parameter_id)
        res = conn.execute(text(sql_update_timestamp))
        logger.info("Updating timestamp: {}".format(res))


os.system("curl --create-dirs -o $HOME/.postgresql/root.crt -O https://cockroachlabs.cloud/clusters/6cadd36b-9892-418c-88c7-64a5781755ec/cert")

dir_path = os.getcwd()

try:
    os.system("apt install -y vim")
    os.system("apt install -y htop")
    os.system("apt install -y libsndfile1")
    os.system("apt install -y unzip")
except:
    logger.warning("You need these libs installed.")


while True:
    logger.info("Get parameter to run.")
    os.environ['DATABASE_URL'] = "cockroachdb://ohh:nZ-eJfpX1fro6l-b9szvCg@free-tier11.gcp-us-east1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&options=--cluster%3Damped-fox-1436"
    os.environ['DATABASE_URL_PSY'] = "postgresql://ohh:nZ-eJfpX1fro6l-b9szvCg@free-tier11.gcp-us-east1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&options=--cluster%3Damped-fox-1436"
    engine = create_engine(os.environ["DATABASE_URL"])
    conn = engine.connect()
    db_df = pd.read_sql("SELECT * FROM parameters_search", con = conn)
    if db_df[pd.isnull(db_df["murmur_file"])].shape[0] == 0:
        logger.info("No parameters to run.")
        sys.exit(0)

    current_timestamp = str(pd.Timestamp.now())

    logger.info("Check if pending running")
    rows_to_fix = db_df[((pd.Timestamp.now() - pd.to_datetime(db_df["start_running_timestamp"])).dt.days >= PATIENCE_DAYS) & (pd.isnull(db_df["mean_murmur"]))]
    if rows_to_fix.shape[0] > 0:
        logger.info("Fix processes without result for more than 48h")
        for row_id, row in rows_to_fix.iterrows():
            update_timestamp(None, row.parameter_id)
            logger.success("Fixing parameters: {}".format(row.parameter_id))
        db_df = pd.read_sql("SELECT * FROM parameters_search", con = conn)

    logger.info("Sampling a parameter")
    parameter_run = db_df.sample().iloc[0]
    logger.info("Update timestamp running")
    db_df.loc[parameter_run.name] = parameter_run
    update_timestamp(current_timestamp, parameter_run.parameter_id)

    
    #Cleaning from previous run    
    os.system("rm -r ./cross-validation-data-1-0-3/")
    os.system("rm -r ./circor-heart-sound/")
    os.system("rm ./murmur_final_result_current.csv")
    os.system("rm ./outcome_final_result_current.csv")
    os.system("rm -r ./circor-heart-sound/")

    os.system("mkdir -p ./cross-validation-data-1-0-3/")
    os.system("mkdir -p ./circor-heart-sound/1.0.3/")
    os.system("git clone --branch matheus https://matheus:{}@github.com/maraujo/physionet22.git".format(github_token))
    assert os.system("pip install -r ./physionet22/requirements.txt") == 0
    assert os.system("pip install tensorflow==2.8.2") == 0

    os.system("rm the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
    assert os.system("wget https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") == 0
    assert os.system("unzip -q -o the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") == 0
    assert os.system("mv ./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data ./circor-heart-sound/1.0.3/") == 0

    os.chdir('physionet22/')
    os.system("python generate_crossvalidation_splits.py")

    logger.info("Saving ohh.config with the following parameters: \n{}".format(pprint.pformat(parameter_run.to_dict())))
    with open("ohh.config", "w") as ohh_config_fpr:
        json.dump(parameter_run.to_dict(), ohh_config_fpr) 

    assert os.system("python ./test_code_crossvalidation_splits.py") == 1

    murmur_df = pd.read_csv("../murmur_final_result_current.csv")
    outcome_df = pd.read_csv("../outcome_final_result_current.csv")
    murmur_mean = murmur_df['Weighted Accuracy'].mean()
    murmur_std = murmur_df['Weighted Accuracy'].std()
    outcome_mean = outcome_df['Cost'].mean()
    outcome_std = outcome_df['Cost'].std()
    update_results(json.dumps(murmur_df.to_dict()), json.dumps(outcome_df.to_dict()), murmur_mean, murmur_std, outcome_std, outcome_mean, parameter_run.parameter_id)

    text_result = ""
    text_result += urllib.parse.quote(open("ohh.config").read() + "\n")
    text_result += urllib.parse.quote(open("../murmur_final_result_current.csv").read() + "\n")
    text_result += urllib.parse.quote(open("../outcome_final_result_current.csv").read() + "\n")
    urllib.request.urlopen("https://vorkqcranza3s6f66wloniatvy0duufg.lambda-url.us-east-1.on.aws/?destiny=matheus.ld.araujo@gmail.com&text={}&password=FIYl4lXi6QHMJHth&subject=DoneLambda".format(text_result))
    os.chdir(dir_path)
