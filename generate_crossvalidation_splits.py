# This scripts create N_FOLDERS of physionet22 data using symbolic links.
from sklearn.model_selection import ShuffleSplit
from helper_code import find_patient_files, load_patient_data, get_label, get_patient_id
import pandas as pd
from pathlib import Path
from glob import glob
import shutil

N_FOLDERS = 10
INPUT_FOLDER = "/physionet_data/challenge/files/circor-heart-sound/1.0.1/training_data/"
OUTPUT_FOLDER = "/physionet_data/challenge/files/cross-validation-data/"
seed = 42
    
def create_folder_and_move(patient_infos, dst_folder):
    #Create folder and copy train files
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    for index, info in patient_infos.iterrows():
        patient_files = glob(INPUT_FOLDER + info["id"] + "*")
        for patient_file in patient_files:
            shutil.copy2(patient_file, dst_folder)
    

if __name__ == "__main__":
    patients_files = find_patient_files(INPUT_FOLDER)
    patient_infos = []

    for patient_file in patients_files:
        patient_data = load_patient_data(patient_file)
        patient_id = get_patient_id(patient_data)
        patient_label = get_label(patient_data)
        print(patient_id)
        patient_infos.append({
            "id" : patient_id,
            "label" : patient_label 
        })
    patient_info_df = pd.DataFrame(patient_infos)

    print("Training Set")
    print("Number of patients : ", patient_info_df.iloc[0])
    print(patient_info_df["label"].value_counts())

    cv = ShuffleSplit(n_splits=N_FOLDERS, test_size=0.2, random_state = seed)
    folder_num = 0

    for train, test in cv.split(patient_info_df):
        print("Folder Number: {}".format(folder_num))
        train_df = patient_info_df.loc[train]
        test_df = patient_info_df.loc[test]
        
        #Create folder and copy train files
        print("Copying train.")
        folder_train = OUTPUT_FOLDER + "{}/train/".format(folder_num)
        create_folder_and_move(train_df, folder_train)
        print("Done.")
        
        #Create folder and move test files
        print("Copying test.")
        folder_test = OUTPUT_FOLDER + "{}/test/".format(folder_num)
        create_folder_and_move(test_df, folder_test)
        print("Done.")
         
        folder_num += 1