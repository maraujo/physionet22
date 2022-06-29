
import os
import json
import boto3
import pandas as pd
  
 
OHH_ARGS = json.loads(open("ohh.config", "r").read().strip())
s3 = boto3.client("s3",  aws_access_key_id=OHH_ARGS["AWS_ID"], aws_secret_access_key=OHH_ARGS["AWS_PASS"])
s3.download_file("1hh-algorithm-dev", '1hh-algorithm-dev/models/hps_search.csv', 'hps_search.csv')

hps_df = pd.read_csv("hps_search.csv")
#find corresponds OHH_ARGS line in hps_search
OHH_ARGS
result = pd.read_csv("model/murmur_result.csv")  
#Update result in corresponding line
hps_df.to_csv("hps_search.csv")
response = s3.upload_file("hps_search.csv", "1hh-algorithm-dev", "models/results/hps_search.csv")
print(response)