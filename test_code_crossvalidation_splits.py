import os
import pandas as pd
commits = ["14cbe8e6b381147622e8f3137db48833193fd700", "05ce59fdd67e2086bcfc47f0c9ee50dbfc6a39fb", "a9827a4dbed3793a5cd32112708b7786d951b159"]
#Run image against crossvalidation
N_FOLDERS = 10
results = []
for commit in commits:
    os.system("git checkout -f {}".format(commit))
    os.system("git status")
    os.system("cp ../evaluate_model.py ./")
    os.system("cp ../test_in_docker.bash ./")
    os.system("docker build -t image .")
    for fold in range(N_FOLDERS):
        print("Commit: {} Fold: {}".format(commit, fold))
        folder = "/physionet_data/challenge/files/cross-validation-data-mini/{}/".format(fold)
        os.system("docker run -v {}model:/physionet/model -v {}/test:/physionet/test_data -v {}/test_outputs:/physionet/test_outputs/ -v {}/train/:/physionet/training_data image bash ./test_in_docker.bash".format(folder,folder,folder,folder))
        script_path = folder + "model/result.csv"
        result = pd.read_csv(script_path)
        result["name"] = commit
        result["fold"] = fold
        results.append(result.iloc[0])
pd.DataFrame(results).to_csv("../final_result.csv")