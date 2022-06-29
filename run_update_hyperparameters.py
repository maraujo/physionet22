
import os
import json
import boto3
import pandas as pd
   
OHH_ARGS = json.loads(open("ohh.config", "r").read().strip())
if ("AWS_ID" in OHH_ARGS) and ("AWS_PASS" in OHH_ARGS):
    s3 = boto3.client("s3",  aws_access_key_id=OHH_ARGS["AWS_ID"], aws_secret_access_key=OHH_ARGS["AWS_PASS"])
    s3.download_file("1hh-algorithm-dev", '1hh-algorithm-dev/models/hps_search.csv', 'hps_search.csv')

hps_df = pd.read_csv("hps_search.csv")
choice_hp = hps_df[pd.isnull(hps_df["auc"])].sample()
OHH_ARGS.update(choice_hp)
json_object = json.dumps(OHH_ARGS, indent = 4) 
ptr = open("ohh.config", "w")
ptr.write(json_object)