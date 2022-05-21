import os
import pandas as pd
commits = ["e288ff0e6d55935c3b2a3679f97205f3ecf75650"]
#Run image against crossvalidation
N_FOLDERS = 10
murmur_results = []
outcome_results = []
RUN_TYPE = "docker" # docker or system
for commit in commits:
    os.system("git checkout -f {}".format(commit))
    os.system("git status")
    os.system("cp ../evaluate_model.py ./")
    os.system("cp ../test_in_docker.bash ./")
    if RUN_TYPE == "docker":
        os.system("docker build -t image .")
    for fold in range(N_FOLDERS):
        print("Commit: {} Fold: {}".format(commit, fold))
        if RUN_TYPE == "docker":
            folder = "/physionet_data/challenge/files/cross-validation-data-1-0-3/{}/".format(fold)
            os.system("docker run -v {}model:/physionet/model -v {}/test:/physionet/test_data -v {}/test_outputs:/physionet/test_outputs/ -v {}/train/:/physionet/training_data image bash ./test_in_docker.bash".format(folder,folder,folder,folder))
        elif RUN_TYPE == "system":
            folder = "./"
            os.system("bash test_code_quick.bash {}".format(fold))
        murmur_script_path = folder + "model/murmur_result.csv"
        outcome_script_path = folder + "model/outcome_result.csv"
        murmur_result = pd.read_csv(murmur_script_path)
        murmur_result["name"] = commit
        murmur_result["fold"] = fold
        murmur_results.append(murmur_result.iloc[0])
        pd.DataFrame(murmur_results).to_csv("../murmur_final_result.csv")
        
        outcome_script_path = folder + "model/outcome_result.csv"
        outcome_result = pd.read_csv(murmur_script_path)
        outcome_result["name"] = commit
        outcome_result["fold"] = fold
        outcome_results.append(outcome_result.iloc[0])
        pd.DataFrame(outcome_results).to_csv("../outcome_final_result.csv")