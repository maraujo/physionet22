import os
#Build Docker Image
# os.system("docker build -t image .")
#Run image against crossvalidation
folder = "/physionet_data/challenge/files/cross-validation-data/0/"
os.system("docker run -v {}model:/physionet/model -v {}/test:/physionet/test_data -v {}/test_outputs:/physionet/test_outputs/ -v {}/train/:/physionet/training_data image bash ./test_in_docker.bash".format(folder,folder,folder,folder))