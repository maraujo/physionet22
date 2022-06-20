import os
import glob
import uuid


github_token = input("Type github token: ")
email_password = input("Matheus email password: ")
ACCESS_KEY = input("Aws ACCESS_KEY: ")
SECRET_KEY = input("Aws SECRET_KEY: ")

os.system("apt install -y vim")
os.system("apt install -y libsndfile1")
os.system("rm -r /physionet_data/challenge/files/cross-validation-data-1-0-3/")

os.system("mkdir -p /physionet_data/challenge/files/cross-validation-data-1-0-3/")
os.system("mkdir -p /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.system("git clone https://matheus:{}@github.com/maraujo/physionet22.git".format(github_token))
os.system("pip install -r ./physionet22/requirements.txt")
os.system("pip install tensorflow==2.8")

import boto3

os.system("rm the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("wget https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip")
os.system("unzip -q -o the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip") 
os.system("mv ./the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data /physionet_data/challenge/files/circor-heart-sound/1.0.3/")
os.chdir('physionet22/')
os.system("python generate_crossvalidation_splits.py")
os.system("python ./test_code_crossvalidation_splits.py &> model_training_output_colab_direct_challenge.txt")
# nohup bash test_code_quick.bash 0 &> model_training_output_colab_direct_challenge.txt
s3 = boto3.client("s3",  aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
lines = []
for filepath in glob.glob("../*.csv"):
    response = s3.upload_file(filepath, "1hh-algorithm-dev", "models/" + str(uuid.uuid4()) + "_" + os.path.basename(filepath))
text = "\n".join(lines)
import urllib
urllib.request.urlopen("https://vorkqcranza3s6f66wloniatvy0duufg.lambda-url.us-east-1.on.aws/?destiny=matheus.ld.araujo@gmail.com&text=DoneLambda&password={}&subject=DoneLambda".format(email_password))