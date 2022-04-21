import os
import pandas as pd
commits = ["a9827a4dbed3793a5cd32112708b7786d951b159", "14cbe8e6b381147622e8f3137db48833193fd700", "05ce59fdd67e2086bcfc47f0c9ee50dbfc6a39fb"]
#Run image against crossvalidation
N_FOLDERS = 10
results = []
for commit in commits:
    # os.system("git checkout {}".format(commit))
    # os.system("cp ../evaluate_model.py ./")
    # os.system("cp ../test_in_docker.bash ./")
    # os.system("docker build -t image .")
    #copy new files
    #build docker
    for fold in range(N_FOLDERS):
        folder = "/physionet_data/challenge/files/cross-validation-data/{}/".format(fold)
        os.system("docker run -v {}model:/physionet/model -v {}/test:/physionet/test_data -v {}/test_outputs:/physionet/test_outputs/ -v {}/train/:/physionet/training_data image bash ./test_in_docker.bash".format(folder,folder,folder,folder))
        script_path = folder + "model/result.csv"
        result = pd.read_csv(script_path)
        result["name"] = commit
        result["fold"] = fold
        results.append(result.iloc[0])
        import ipdb;ipdb.set_trace()
        pass