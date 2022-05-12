import os
import pandas as pd
commits = ["b8501999a67f2a486da7d644d4c0bacb677b61be"]
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
        folder = "/physionet_data/challenge/files/cross-validation-data-1-0-3/{}/".format(fold)
        os.system("docker run -v {}model:/physionet/model -v {}/test:/physionet/test_data -v {}/test_outputs:/physionet/test_outputs/ -v {}/train/:/physionet/training_data image bash ./test_in_docker.bash".format(folder,folder,folder,folder))
        script_path = folder + "model/result.csv"
        result = pd.read_csv(script_path)
        result["name"] = commit
        result["fold"] = fold
        results.append(result.iloc[0])
        pd.DataFrame(results).to_csv("../final_result.csv")