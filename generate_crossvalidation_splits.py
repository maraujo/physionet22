# This scripts create N_FOLDERS of physionet22 data using symbolic links.
from sklearn.model_selection import ShuffleSplit
from helper_code import find_patient_files, load_patient_data, get_murmur, get_patient_id
import pandas as pd
from pathlib import Path
from glob import glob
import shutil

N_FOLDERS = 5
print("N_FOLDERS: {}".format(N_FOLDERS))
INPUT_FOLDER = "../circor-heart-sound/1.0.3/training_data/"
IS_MINI = False
if IS_MINI:
    OUTPUT_FOLDER = "../cross-validation-data-1-0-3-mini/"
else:
    OUTPUT_FOLDER = "../cross-validation-data-1-0-3/"
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
        patient_label = get_murmur(patient_data)
        patient_infos.append({
            "id" : patient_id,
            "label" : patient_label 
        })
    if IS_MINI:
        patient_info_df = pd.DataFrame(patient_infos).sample(50, random_state=42)
    else:
        patient_info_df = pd.DataFrame(patient_infos).sample(frac=1, random_state=42)
    
    print("Training Set")
    print("Number of patients : ", patient_info_df.iloc[0])
    print(patient_info_df["label"].value_counts())

    cv = ShuffleSplit(n_splits=N_FOLDERS, test_size=0.3, random_state = seed)
    folder_num = 0

    for train, test in cv.split(patient_info_df):
        print("Folder Number: {}".format(folder_num))
        train_df = patient_info_df.iloc[train]
        test_df = patient_info_df.iloc[test]
        
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