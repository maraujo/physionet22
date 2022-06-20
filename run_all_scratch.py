import os
import glob


github_token = input("Type github token: ")
email_password = input("Matheus email password: ")
ACCESS_KEY = input("Aws ACCESS_KEY: ")
SECRET_KEY = input("Aws SECRET_KEY: ")

os.system("rm -r /physionet_data/challenge/files/cross-validation-data-1-0-3/")

os.system("mkdir -p /physionet_data/challenge/files/cross-validation-data-1-0-3/")
os.system("mkdir -p /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.system("git clone https://matheus:{}@github.com/maraujo/physionet22.git".format(github_token))
os.system("pip install -r /content/physionet22/requirements.txt")
os.system("pip install tensorflow==2.8")

import boto3

os.system("rm the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("wget https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("unzip -q -o the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") 
os.system("mv /content/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.chdir('physionet22/')
os.system("python generate_crossvalidation_splits.py")
os.system("nohup python /content/physionet22/test_code_crossvalidation_splits.py &> model_training_output_colab_direct_challenge.txt")
# nohup bash test_code_quick.bash 0 &> model_training_output_colab_direct_challenge.txt
s3 = boto3.client("s3",  aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
lines = []
for filepath in glob.glob("../*.csv"):
    bucket = s3.Bucket("hh-algorithm-dev")
    response = bucket.upload_file(filepath, "models/" + os.path.basename(filepath))
text = "\n".join(lines)
import urllib
urllib.request.urlopen("https://vorkqcranza3s6f66wloniatvy0duufg.lambda-url.us-east-1.on.aws/?destiny=matheus.ld.araujo@gmail.com&text=DoneLambda&password={}&subject=DoneLambda".format(email_password))