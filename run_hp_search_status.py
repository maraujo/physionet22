import os
import glob
import uuid
from loguru import logger
import os
from sqlalchemy import create_engine, text, update
import pandas as pd
import sys
import logging
import psycopg2
import json
import pprint
from tabulate import tabulate

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

logger.info("Get parameter to run.")
os.system("curl --create-dirs -o $HOME/.postgresql/root.crt -O https://cockroachlabs.cloud/clusters/6cadd36b-9892-418c-88c7-64a5781755ec/cert")
os.environ['DATABASE_URL'] = "cockroachdb://ohh:nZ-eJfpX1fro6l-b9szvCg@free-tier11.gcp-us-east1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&options=--cluster%3Damped-fox-1436"
os.environ['DATABASE_URL_PSY'] = "postgresql://ohh:nZ-eJfpX1fro6l-b9szvCg@free-tier11.gcp-us-east1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&options=--cluster%3Damped-fox-1436"
engine = create_engine(os.environ["DATABASE_URL"])
conn = engine.connect()
db_df = pd.read_sql("SELECT * FROM parameters_search", con = conn)

logger.info("Completed: {}/{}".format(db_df[~pd.isnull(db_df["mean_murmur"])].shape[0], db_df.shape[0]))
logger.info("Processing now: {}".format(db_df[~pd.isnull(db_df["start_running_timestamp"]) & pd.isnull(db_df["mean_murmur"])].shape[0]))

logger.info("Top 50 Murmurs:\n{}".format(tabulate(db_df.drop(["murmur_file", "outcome_file"],axis=1).sort_values(by="mean_murmur", ascending=False).head(20))))
