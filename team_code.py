#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import openl3
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

def get_embs_df_from_patient_data(main_model, num_patient_files, patient_files, data_folder, verbose):
    patients_embs = []
    for i in range(num_patient_files):
        if i > 12:
            break
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        recordings, frequencies = load_recordings(data_folder, current_patient_data, get_frequencies=True) 

        #Extract Embeedings
        embs_part, tss_part = openl3.get_audio_embedding(recordings, frequencies, hop_size=SECONDS_PER_EMBEDDING, batch_size=4,   verbose=1 , model=main_model)

        #Prepare Features DataFrame
        record_files = []
        audios_embs = []
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            filename = entries[2]
            filepath = os.path.join(data_folder, filename)
            recording_pos = recording_information[i].split(" ")[0]
            audio_embs_df = pd.DataFrame(embs_part[i])
            audio_embs_df[POS] = recording_pos
            audio_embs_df[TIME] = tss_part[i]
            audios_embs.append(audio_embs_df)
        audios_embs_df = pd.concat(audios_embs)
        audios_embs_df = audios_embs_df.reset_index(drop=True)

        # If murmur present, remove places where it is not audible (not in #Murmur locations)
        patient_data_series = pd.Series(current_patient_data.split('\n')[num_locations+1:])
        murmur_locations = patient_data_series[patient_data_series.str.contains("Murmur location")].iloc[0].split(": ")[-1]
        murmur_locations = murmur_locations.split("+")
        for location in murmur_locations:
            audios_embs_df = audios_embs_df[audios_embs_df[POS] != location]
        
        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1
        audios_embs_df[classes[0]] = current_labels[0]
        audios_embs_df[classes[1]] = current_labels[1]
        audios_embs_df[classes[2]] = current_labels[2]
        audios_embs_df[ID] =  current_patient_data.split(" ")[0]
        patients_embs.append(audios_embs_df)
    all_patients_embs_df = pd.concat(patients_embs)
    return all_patients_embs_df

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=512)
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
    
    if False and os.path.exists(TEMP_FILE):
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
    else:
        all_patients_embs_df = get_embs_df_from_patient_data(main_model, num_patient_files, patient_files, data_folder, verbose)
        all_patients_embs_df.to_pickle(TEMP_FILE)

    #Stratify Sample
    X_train, X_test, y_train, y_test = train_test_split(all_patients_embs_df[range(0,512)].values, all_patients_embs_df[classes].values, test_size=0.2, random_state=42, stratify= all_patients_embs_df[classes].values.argmax(axis=1))
    

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators = 100    # Number of trees in the forest.
    max_leaf_nodes = 100 # Maximum number of leaf nodes in each tree.
    random_state = 42   # Random state; set for reproducibility.

    classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(X_train, y_train)
    print(classifier.score(X_test, y_test))
    # Save the model.
    save_challenge_model(model_folder, classes, classifier)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = model['classes']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, classes, classifier):
    d = {'classes': classes, 'classifier': classifier}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1

    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)

    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Extract recording locations and data. Identify when a location is present, and compute the mean, variance, and skewness of
    # each recording. If there are multiple recordings for one location, then extract features from the last recording.
    locations = get_locations(data)

    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    if num_locations==num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i])>0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = np.var(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])
    recording_features = recording_features.flatten()

    features = np.hstack(([age], sex_features, [height], [weight], [is_pregnant], recording_features))

    return np.asarray(features, dtype=np.float32)
