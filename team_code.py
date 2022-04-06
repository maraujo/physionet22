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
import torch
import torch.nn.functional as F
import pandas as pd
from resnet import ResNet1d
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
SEQ_LENGTH = 24576

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # do nothing, we just want to test our pretrained model
    return 0

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    # define model
    model = ResNet1d(
        input_dim=(1,SEQ_LENGTH), 
        blocks_dim=list(zip([64, 128, 196, 256, 320], [4096, 1024, 256, 64, 16])), 
        n_classes=3, 
        kernel_size=17, 
        dropout_rate=0.8,
    )
    # load the pretrained model
    model_dict = torch.load(os.path.join(model_folder, 'model.pth'), map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval()
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    classes = ['Present', 'Unknown', 'Absent']
    all_probabilities = []
    all_labels = []
    # do prediction for each recording
    if verbose:
        patient_id = get_patient_id(data)
        print(f'processing patient_id:{patient_id}')
    with torch.no_grad():
        for recording in recordings:
            # normalized recording
            recording_mean = recording.mean()
            recording_std = recording.std()
            recording = (recording - recording_mean) / recording_std
            # now, let's just random pick one segment from the recording to make prediction
            current_recording_length = recording.shape[0]
            if current_recording_length > SEQ_LENGTH:
                start_point = 0
                steps = []
                while (start_point + SEQ_LENGTH) < current_recording_length:
                    steps += [start_point]
                    start_point += SEQ_LENGTH # non-overlap
                steps += [current_recording_length - SEQ_LENGTH]
                # print(f'steps:{steps}')
                recording_segments = []
                for step in steps:
                    recording_segments.append(recording[step:step+SEQ_LENGTH])
                recording_segment = torch.from_numpy(np.stack(recording_segments)[:,np.newaxis,:]).float()
            else: # padding with zeros
                recording_segment = np.zeros(SEQ_LENGTH)
                recording_segment[0:current_recording_length] = recording
                recording_segment = torch.from_numpy(recording_segment[None,None]).float()
            # make prediction
            out = model(recording_segment)
            out_softmax = F.softmax(out, dim=1)
            out_argmax = torch.argmax(out_softmax, dim=1)
            # if there are multiple segments in this recording, the final prediction is the one with the largest "Present" probability
            out_softmax_0 = out_softmax[:,0]
            max_0_index = torch.argmax(out_softmax_0, dim=0)
            all_probabilities.append(out_softmax[max_0_index].detach().cpu().numpy())
            all_labels.append(out_argmax[max_0_index].detach().cpu().numpy())
    labels = to_one_hot_bool(min(all_labels), len(classes))
    probabilities = all_probabilities[all_labels.index(min(all_labels))]
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

def to_one_hot_bool(data, num_classes):
    """
    convert a numpy array to one hot bool
    data: [N] the array that needs to be converted
    num_classes: how many classes in this array
    """
    result = np.eye(num_classes)[data].astype(np.bool_)
    return result