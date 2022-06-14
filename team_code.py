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
from keras.models import load_model
from urllib.request import urlopen, Request
from urllib.parse import urlparse 
import urllib.request

from tqdm import tqdm
from zipfile import ZipFile
from loguru import logger
from glob import glob
from autokeras.keras_layers import CastToFloat32
from sklearn.preprocessing import OneHotEncoder


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
LOAD_TRAINED_MODELS = True


class DownloadProgressBar(tqdm):
    # From https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)



#Download autoencoder
enc = OneHotEncoder()
enc.fit([[True], [False]])
        
def download_model_from_url(url, file_name):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = file_name
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url(url, cached_file)
 
 
def compute_confusion_matrix(labels, outputs):
    assert(np.shape(labels)[0] == np.shape(outputs)[0])
    assert all(value in (0, 1, True, False) for value in np.unique(labels)), np.unique(labels) 
    assert all(value in (0, 1, True, False) for value in np.unique(outputs)), np.unique(outputs)

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A

# Compute accuracy.
def compute_weighted_accuracy(labels, outputs, classes = ["Abnormal", "Normal"]):
    # Define constants.
    if classes == ['Present', 'Unknown', 'Absent']:
        weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    elif classes == ['Abnormal', 'Normal']:
        weights = np.array([[5, 1], [5, 1]])
    else:
        raise NotImplementedError('Weighted accuracy undefined for classes {}'.format(', '.join(classes)))
    # Compute confusion matrix.
    assert(np.shape(labels) == np.shape(outputs))
    outputs = outputs > 0.5
    outputs = outputs.numpy()

    labels = labels == 1
    labels = labels.numpy()
    
    
    labels = enc.transform(labels).toarray()
    outputs = enc.transform(outputs).toarray()

    A = compute_confusion_matrix(labels, outputs)
    
    # Multiply the confusion matrix by the weight matrix.
    assert(np.shape(A) == np.shape(weights))
    B = weights * A

    # Compute weighted_accuracy.
    if np.sum(B) > 0:
        weighted_accuracy = np.trace(B) / np.sum(B)
    else:
        weighted_accuracy = float('nan')

    return weighted_accuracy       
        
def load_pretrained_model(model_dir, model_type):
    logger.info("Loading model: {}".format(model_type))
    os.makedirs(model_dir, exist_ok=True)
    if model_type == "murmur":
        destiny_zip = os.path.join(model_dir, 'murmur_model.zip')
        destiny_model = os.path.join(model_dir, 'murmur_model.model')
        url = "http://algodev.matheusaraujo.com:8888/pretrained_models/model_murmur_checkpoint72_x_72.zip"
        
    if model_type == "noise":
        destiny_zip = os.path.join(model_dir, 'noise_model.zip')
        destiny_model = os.path.join(model_dir, 'noise_model.model')
        url = "http://algodev.matheusaraujo.com:8888/pretrained_models/noise_model_108x108.zip"
        
    if model_type == "murmur_decision":
        destiny_zip = os.path.join(model_dir, 'murmur_decision.zip')
        destiny_model = os.path.join(model_dir, 'murmur_decision.model')
        url = "http://algodev.matheusaraujo.com:8888/pretrained_models/murmur_decision_best_model-15_samples_64_embs.zip"

    download_model_from_url(url, destiny_zip)
    with ZipFile(destiny_zip, 'r') as zipObj:
    # Extract all the contents of zip file in current directory
        zipObj.extractall(destiny_model)
    keras_model_folder_to_load = glob(destiny_model + os.path.sep + "*")[0]
    return load_model(keras_model_folder_to_load, custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
    

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    
    if LOAD_TRAINED_MODELS:
        noise_model = load_pretrained_model(model_folder, "noise")
        murmur_model = load_pretrained_model(model_folder, "murmur")
        murmur_decision_model = load_pretrained_model(model_folder, "murmur_decision")
    
    import ipdb;ipdb.set_trace()
    pass
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()

    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)

        # Extract features.
        current_features = get_features(current_patient_data, current_recordings)
        features.append(current_features)

        # Extract labels and use one-hot encoding.
        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)

    features = np.vstack(features)
    murmurs = np.vstack(murmurs)
    outcomes = np.vstack(outcomes)

    # Train the model.
    if verbose >= 1:
        print('Training model...')

    # Define parameters for random forest classifier.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 45   # Maximum number of leaf nodes in each tree.
    random_state   = 6789 # Random state; set for reproducibility.

    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, murmurs)
    outcome_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes)

    # Save the model.
    save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier)

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
    imputer = model['imputer']
    murmur_classes = model['murmur_classes']
    murmur_classifier = model['murmur_classifier']
    outcome_classes = model['outcome_classes']
    outcome_classifier = model['outcome_classifier']

    # Load features.
    features = get_features(data, recordings)

    # Impute missing data.
    features = features.reshape(1, -1)
    features = imputer.transform(features)

    # Get classifier probabilities.
    murmur_probabilities = murmur_classifier.predict_proba(features)
    murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)[:, 0, 1]
    outcome_probabilities = outcome_classifier.predict_proba(features)
    outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)[:, 0, 1]

    # Choose label with highest probability.
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    idx = np.argmax(murmur_probabilities)
    murmur_labels[idx] = 1
    outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    idx = np.argmax(outcome_probabilities)
    outcome_labels[idx] = 1

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, murmur_classes, murmur_classifier, outcome_classes, outcome_classifier):
    d = {'imputer': imputer, 'murmur_classes': murmur_classes, 'murmur_classifier': murmur_classifier, 'outcome_classes': outcome_classes, 'outcome_classifier': outcome_classifier}
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