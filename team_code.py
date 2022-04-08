#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import shutil
from pathlib import Path
from helper_code import *
<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np, pandas as pd, scipy as sp, scipy.stats, os, sys, joblib
import openl3

=======
=======
>>>>>>> 138735d (merging dewen in master)
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from resnet import ResNet1d
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
>>>>>>> ad44dd9 (change to download from url)
=======
=======
import numpy as np, pandas as pd, scipy as sp, scipy.stats, os, sys, joblib
import openl3

>>>>>>> da293e5 (Implemented a lightgbm version with SMOTE)
>>>>>>> 138735d (merging dewen in master)
from sklearn.metrics import classification_report

from urllib.error import HTTPError
from urllib.request import urlopen, Request
from urllib.parse import urlparse 

import hashlib
import shutil
import tempfile
from tqdm import tqdm

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################
<<<<<<< HEAD

from pycaret.classification import create_model, finalize_model, setup, predict_model, pull, save_model
from pycaret.classification import load_config, save_config, load_model

POS = "position"
TIME = "time"
LABEL = "label"
ID = "ID"
TEMP_FILE = "/tmp/cache_challege_df.pkl"
SECONDS_PER_EMBEDDING = 2
classes = ['Present', 'Unknown', 'Absent']
num_classes = len(classes)
main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=512)
TRAIN_SMALL_SAMPLE = False
submission = True

def get_embs_df_from_patient_data(num_patient_files, patient_files, data_folder, verbose):
    patients_embs = []
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        if TRAIN_SMALL_SAMPLE and i > 50:
            break

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        recordings, frequencies = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        pid = current_patient_data.split(" ")[0]

        # Debugging:
        # print("PID:", pid)
        # if current_patient_data.split(" ")[0] != "45843":
        #     continue

        # Get where murmur is for this patient
        patient_data_series = pd.Series(current_patient_data.split('\n')[num_locations+1:])
        murmur_locations = patient_data_series[patient_data_series.str.contains("Murmur location")].iloc[0].split(": ")[-1]
        murmur_locations = murmur_locations.split("+")

        #Extract Embeedings
        embs_part, tss_part = openl3.get_audio_embedding(recordings, frequencies, hop_size=SECONDS_PER_EMBEDDING, batch_size=4, verbose=1, model=main_model)

        #Prepare Features DataFrame
        record_files = []
        audios_embs = []
        for j in range(num_locations):
            entries = recording_information[j].split(' ')
            recording_pos = entries[0]
            # filename = entries[2]
            # filepath = os.path.join(data_folder, filename)
            audio_embs_df = pd.DataFrame(embs_part[j])
            audio_embs_df[POS] = recording_pos
            audio_embs_df[TIME] = tss_part[j]
            audio_embs_df["has_murmur"] = recording_pos in murmur_locations
            audios_embs.append(audio_embs_df)

        audios_embs_df = pd.concat(audios_embs)
        audios_embs_df = audios_embs_df.reset_index(drop=True)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1

        audios_embs_df[classes[0]] = current_labels[0]
        audios_embs_df[classes[1]] = current_labels[1]
        audios_embs_df[classes[2]] = current_labels[2]
        audios_embs_df[ID] = pid

        patients_embs.append(audios_embs_df)

    all_patients_embs_df = pd.concat(patients_embs)
    return all_patients_embs_df


def copy_test_data(test_ids, source, dest):

    # Removes everything in the dest folder
    [f.unlink() for f in Path(dest).glob("*") if f.is_file()]

    # Copies selected files from the source to the dest folder
    for pid in test_ids:
        for f in Path(source).glob(pid + "*"):
            shutil.copyfile(f, os.path.join(dest, os.path.basename(f)))

=======
<<<<<<< HEAD
SEQ_LENGTH = 24576
>>>>>>> 138735d (merging dewen in master)

# Model definition
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y

class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x
=======

from pycaret.classification import *
from pycaret.classification import load_config

POS = "position"
TIME = "time"
LABEL = "label"
ID = "ID"
TEMP_FILE = "/tmp/cache_challege_df.pkl"
SECONDS_PER_EMBEDDING = 2
classes = ['Present', 'Unknown', 'Absent']
num_classes = len(classes)
main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=512)
TRAIN_SMALL_SAMPLE = False

def get_embs_df_from_patient_data(num_patient_files, patient_files, data_folder, verbose):
    patients_embs = []
    for i in range(num_patient_files):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patient_files))

        if TRAIN_SMALL_SAMPLE and i > 50:
            break

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        recordings, frequencies = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        pid = current_patient_data.split(" ")[0]

        # Debugging:
        # print("PID:", pid)
        # if current_patient_data.split(" ")[0] != "45843":
        #     continue

        # Get where murmur is for this patient
        patient_data_series = pd.Series(current_patient_data.split('\n')[num_locations+1:])
        murmur_locations = patient_data_series[patient_data_series.str.contains("Murmur location")].iloc[0].split(": ")[-1]
        murmur_locations = murmur_locations.split("+")

        #Extract Embeedings
        embs_part, tss_part = openl3.get_audio_embedding(recordings, frequencies, hop_size=SECONDS_PER_EMBEDDING, batch_size=4, verbose=1, model=main_model)

        #Prepare Features DataFrame
        record_files = []
        audios_embs = []
        for j in range(num_locations):
            entries = recording_information[j].split(' ')
            recording_pos = entries[0]
            # filename = entries[2]
            # filepath = os.path.join(data_folder, filename)
            audio_embs_df = pd.DataFrame(embs_part[j])
            audio_embs_df[POS] = recording_pos
            audio_embs_df[TIME] = tss_part[j]
            audio_embs_df["has_murmur"] = recording_pos in murmur_locations
            audios_embs.append(audio_embs_df)

        audios_embs_df = pd.concat(audios_embs)
        audios_embs_df = audios_embs_df.reset_index(drop=True)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(num_classes, dtype=int)
        label = get_label(current_patient_data)
        if label in classes:
            j = classes.index(label)
            current_labels[j] = 1

        audios_embs_df[classes[0]] = current_labels[0]
        audios_embs_df[classes[1]] = current_labels[1]
        audios_embs_df[classes[2]] = current_labels[2]
        audios_embs_df[ID] = pid

        patients_embs.append(audios_embs_df)

    all_patients_embs_df = pd.concat(patients_embs)
    return all_patients_embs_df
>>>>>>> da293e5 (Implemented a lightgbm version with SMOTE)


def copy_test_data(test_ids, source, dest):

    # Removes everything in the dest folder
    [f.unlink() for f in Path(dest).glob("*") if f.is_file()]

    # Copies selected files from the source to the dest folder
    for pid in test_ids:
        for f in Path(source).glob(pid + "*"):
            shutil.copyfile(f, os.path.join(dest, os.path.basename(f)))


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
<<<<<<< HEAD
<<<<<<< HEAD
    # main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=512)

    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    if os.path.exists(TEMP_FILE):
        print(" ----------------------------   Loading TEMP file! ------------------------------------------------")
        print(TEMP_FILE)
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
    else:
        all_patients_embs_df = get_embs_df_from_patient_data(num_patient_files, patient_files, data_folder, verbose)
        try:
            all_patients_embs_df.to_pickle(TEMP_FILE)
        except:
            print("Could not save TEMP file")

    #Stratify Sample
    all_patients_embs_df = all_patients_embs_df.reset_index(drop=True)
    ######
    print("Running Pycaret:")

    # What should we do with Unknown (?)
    UNK_OPT = "positive"

    if UNK_OPT == "positive":
        all_patients_embs_df.loc[all_patients_embs_df["Unknown"] == True, "has_murmur"] = True   # We could assume that unknown is positive label
    elif UNK_OPT == "remove":
        all_patients_embs_df = all_patients_embs_df[all_patients_embs_df["Unknown"] != True]  # Or we could just drop these instances
    else:
        pass

    features_to_ignore = ["Present", "Unknown", "Absent", "ID", "time", "position", "has_murmur"]
    all_patients_embs_df["label"] = all_patients_embs_df["has_murmur"]

    seed = 42
    np.random.seed(seed)

    pids = all_patients_embs_df["ID"].unique()
    np.random.shuffle(pids)

    train_ids = set(pids[:int(pids.shape[0]*.8)])
    test_ids = set(pids) - train_ids

    train_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(train_ids)]
    test_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(test_ids)]

    exp = setup(data=train_data, test_data=test_data,
                target='label', ignore_features=features_to_ignore, silent=True,
                remove_perfect_collinearity=True, fold=2,
                normalize=True, normalize_method="robust",
                fix_imbalance=True,
                fold_strategy="groupkfold", fold_groups="ID",
                use_gpu=False, session_id=seed)

    if not submission:
        copy_test_data(test_ids, "/home/palotti/github/physionet22/all_training_data/", "/home/palotti/github/physionet22/test_data/")

    # top3 = compare_models(include=["et", "rf", "lda", "lr", "ridge", "lightgbm", "gbc", "xgboost"],  n_select=3)
    # top3 = compare_models(include=["lda", "lr", "ridge", "lightgbm", "dummy"], n_select=3, sort='F1')
    # top3 = compare_models(include=["et", "rf", "lda", "lr", "ridge", "lightgbm"], n_select=3)
    # classifier = compare_models(include=["rf"], n_select=1)

    # rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42,)
    # classifier = create_model(rf)
    # classifier = create_model("rf", n_estimators=100, max_leaf_nodes=100, random_state=42)
    # print("TOP 3 Classifiers")
    # for c in top3:
    #     print(c)
    # print("----------------------------")

    # classifier = create_model("lightgbm")
    # classifier = tune_model(classifier, choose_better=True)
    # classifier = blend_models(top3, weights=[1,1,10])
    # classifier = top3[0]

    # classifier = create_model("gbc", ccp_alpha=0.0, criterion='friedman_mse', init=None,
    #                    learning_rate=0.1, loss='deviance', max_depth=3,
    #                    max_features=None, max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=1, min_samples_split=2,
    #                    min_weight_fraction_leaf=0.0, n_estimators=100,
    #                    n_iter_no_change=None, presort='deprecated',
    #                    random_state=42, subsample=1.0, tol=0.0001,
    #                    validation_fraction=0.1, warm_start=False, cross_validation=False)  # Optimized elsewhere
    # classifier = create_model("ridge", cross_validation=False)
    # classifier = tune_model(classifier, n_iter=100, optimize="Recall")

    classifier = create_model("lightgbm", )
    print("Starts with:", classifier)
    # classifier = tune_model(classifier, n_iter=200, optimize="F1", choose_better=True) # TODO: after many hours, it returned the same initial classifier

    # classifier = create_model("ridge")
    if verbose >= 0:

        print("Best Model:", classifier)
        predict_model(classifier)
        print("Test results:")
        df_results = pull()
        for col in df_results.iteritems():
            print(col[0], ":", col[1].values[0])

    # print(plot_model(classifier, plot='confusion_matrix'))
    # print(classification_report(y_test, classifier.predict(X_test)))

    #Check primarily results
    #    print(classification_report(y_test, classifier.predict(X_test)))

    if submission:
        finalize_model(classifier)  # This should only be used when submitting it to the challenge

    # Save the model.
    save_challenge_model(model_folder, classes, classifier)

    if verbose >= 1:
        print('Done.')


def run_challenge_model(model, data, recordings, verbose, thresholds=None, use_cache=False):
=======
=======
>>>>>>> 138735d (merging dewen in master)
    # download the pretrained model
    download_model_from_url('https://www3.nd.edu/~scl/models/model.pth',model_folder,'model.pth')
    return 0
>>>>>>> ad44dd9 (change to download from url)

    classes = ['Present', 'Unknown', 'Absent']
<<<<<<< HEAD
    classifier = model # ['classifier']
    fields = data.split("\n")[0].split(" ")

    pid = fields[0]
    frequency_sample_rate = int(fields[2])

    # Debugging
    # if pid != "85108":
    #     return classes,  [1,0,0],  [1,0,0]

    if use_cache:
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
        df_input = all_patients_embs_df[all_patients_embs_df["ID"] == str(pid)]

    else:
        embs_part, _ = openl3.get_audio_embedding(recordings, [frequency_sample_rate] * len(recordings), hop_size=SECONDS_PER_EMBEDDING, batch_size=4, verbose=1, model=main_model)

        df_input = [ ]
        for i, part in enumerate(embs_part):
            df_tmp = pd.DataFrame(part)
            df_tmp["position"] = i
            df_input.append(df_tmp)
        df_input = pd.concat(df_input)


    df_input = predict_model(classifier, data=df_input)
    df_input["Label"] = df_input["Label"].apply(lambda x: 1 if x == "True" else 0) # This weird line is meant to convert the strings "True"/"False" into 1/0

    predictions = df_input[["position", "Label"]].groupby("position")["Label"].mean()

    if thresholds is None:
        thresholds = {"min1": 0.01, "n1": 1, "min2": 0.01, "n2": 1}

    if (predictions >= thresholds["min1"]).sum() >= thresholds["n1"]:
        labels = [1, 0, 0]
        if verbose >=0 and "Present" not in data:
            print("Error marked as Present. Pid:", pid)

    elif (predictions >= thresholds["min2"]).sum() >= thresholds["n2"]:
        if verbose >=0 and "Present" in data:
            print("Error marked as Unk. Pid:", pid)
        labels = [0, 1, 0]
    else:
        if verbose >=0 and "Present" in data:
            print("Error marked as Absent. Pid:", pid)
        labels = [0, 0, 1]

    probabilities = labels
=======
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
            current_recording_length = recording.shape[0]
            # divide the whole recording into non-overlap segments, and predict each segment
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
<<<<<<< HEAD
>>>>>>> ad44dd9 (change to download from url)
=======
=======
    # main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=512)

    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    if os.path.exists(TEMP_FILE):
        print(" ----------------------------   Loading TEMP file! ------------------------------------------------")
        print(TEMP_FILE)
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
    else:
        all_patients_embs_df = get_embs_df_from_patient_data(num_patient_files, patient_files, data_folder, verbose)
        try:
            all_patients_embs_df.to_pickle(TEMP_FILE)
        except:
            print("Could not save TEMP file")

    #Stratify Sample
    all_patients_embs_df = all_patients_embs_df.reset_index(drop=True)
    ######
    print("Running Pycaret:")

    # What should we do with Unknown (?)
    UNK_OPT = "positive"

    if UNK_OPT == "positive":
        all_patients_embs_df.loc[all_patients_embs_df["Unknown"] == True, "has_murmur"] = True   # We could assume that unknown is positive label
    elif UNK_OPT == "remove":
        all_patients_embs_df = all_patients_embs_df[all_patients_embs_df["Unknown"] != True]  # Or we could just drop these instances
    else:
        pass

    features_to_ignore = ["Present", "Unknown", "Absent", "ID", "time", "position", "has_murmur"]
    all_patients_embs_df["label"] = all_patients_embs_df["has_murmur"]

    seed = 42
    np.random.seed(seed)

    pids = all_patients_embs_df["ID"].unique()
    np.random.shuffle(pids)

    train_ids = set(pids[:int(pids.shape[0]*.8)])
    test_ids = set(pids) - train_ids

    train_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(train_ids)]
    test_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(test_ids)]

    exp = setup(data=train_data, test_data=test_data,
                target='label', ignore_features=features_to_ignore, silent=True,
                remove_perfect_collinearity=True,
                normalize=True, normalize_method="robust",
                fix_imbalance=True,
                fold_strategy="groupkfold", fold_groups="ID",
                use_gpu=False, session_id=seed)

    if not submission:
        copy_test_data(test_ids, "/home/palotti/github/physionet22/all_training_data/", "/home/palotti/github/physionet22/test_data/")

    # top3 = compare_models(include=["et", "rf", "lda", "lr", "ridge", "lightgbm", "gbc", "xgboost"],  n_select=3)
    # top3 = compare_models(include=["lda", "lr", "ridge", "lightgbm", "dummy"], n_select=3, sort='F1')
    # top3 = compare_models(include=["et", "rf", "lda", "lr", "ridge", "lightgbm"], n_select=3)
    # classifier = compare_models(include=["rf"], n_select=1)

    # rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42,)
    # classifier = create_model(rf)
    # classifier = create_model("rf", n_estimators=100, max_leaf_nodes=100, random_state=42)
    # print("TOP 3 Classifiers")
    # for c in top3:
    #     print(c)
    # print("----------------------------")

    # classifier = create_model("lightgbm")
    # classifier = tune_model(classifier, choose_better=True)
    # classifier = blend_models(top3, weights=[1,1,10])
    # classifier = top3[0]

    # classifier = create_model("gbc", ccp_alpha=0.0, criterion='friedman_mse', init=None,
    #                    learning_rate=0.1, loss='deviance', max_depth=3,
    #                    max_features=None, max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=1, min_samples_split=2,
    #                    min_weight_fraction_leaf=0.0, n_estimators=100,
    #                    n_iter_no_change=None, presort='deprecated',
    #                    random_state=42, subsample=1.0, tol=0.0001,
    #                    validation_fraction=0.1, warm_start=False, cross_validation=False)  # Optimized elsewhere
    # classifier = create_model("ridge", cross_validation=False)
    # classifier = tune_model(classifier, n_iter=100, optimize="Recall")

    classifier = create_model("lightgbm")
    print("Starts with:", classifier)
    # classifier = tune_model(classifier, n_iter=200, optimize="F1", choose_better=True) # TODO: after many hours, it returned the same initial classifier

    # classifier = create_model("ridge")
    if verbose >= 0:

        print("Best Model:", classifier)
        predict_model(classifier)
        print("Test results:")
        df_results = pull()
        for col in df_results.iteritems():
            print(col[0], ":", col[1].values[0])

    # print(plot_model(classifier, plot='confusion_matrix'))
    # print(classification_report(y_test, classifier.predict(X_test)))

    #Check primarily results
    #    print(classification_report(y_test, classifier.predict(X_test)))

    if submission:
        finalize_model(classifier)  # This should only be used when submitting it to the challenge

    # Save the model.
    save_challenge_model(model_folder, classes, classifier)

    if verbose >= 1:
        print('Done.')


def run_challenge_model(model, data, recordings, verbose, thresholds=None, use_cache=False):

    classes = ['Present', 'Unknown', 'Absent']
    classifier = model # ['classifier']
    fields = data.split("\n")[0].split(" ")

    pid = fields[0]
    frequency_sample_rate = int(fields[2])

    # Debugging
    # if pid != "85108":
    #     return classes,  [1,0,0],  [1,0,0]

    if use_cache:
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
        df_input = all_patients_embs_df[all_patients_embs_df["ID"] == str(pid)]

    else:
        embs_part, _ = openl3.get_audio_embedding(recordings, [frequency_sample_rate] * len(recordings), hop_size=SECONDS_PER_EMBEDDING, batch_size=4, verbose=1, model=main_model)

        df_input = [ ]
        for i, part in enumerate(embs_part):
            df_tmp = pd.DataFrame(part)
            df_tmp["position"] = i
            df_input.append(df_tmp)
        df_input = pd.concat(df_input)


    df_input = predict_model(classifier, data=df_input)
    df_input["Label"] = df_input["Label"].apply(lambda x: 1 if x == "True" else 0) # This weird line is meant to convert the strings "True"/"False" into 1/0

    predictions = df_input[["position", "Label"]].groupby("position")["Label"].mean()

    if thresholds is None:
        thresholds = {"min1": 0.01, "n1": 1, "min2": 0.01, "n2": 1}

    if (predictions >= thresholds["min1"]).sum() >= thresholds["n1"]:
        labels = [1, 0, 0]
        if verbose >=0 and "Present" not in data:
            print("Error marked as Present. Pid:", pid)

    elif (predictions >= thresholds["min2"]).sum() >= thresholds["n2"]:
        if verbose >=0 and "Present" in data:
            print("Error marked as Unk. Pid:", pid)
        labels = [0, 1, 0]
    else:
        if verbose >=0 and "Present" in data:
            print("Error marked as Absent. Pid:", pid)
        labels = [0, 0, 1]

    probabilities = labels
>>>>>>> da293e5 (Implemented a lightgbm version with SMOTE)
>>>>>>> 138735d (merging dewen in master)
    return classes, labels, probabilities


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    load_config(os.path.join(model_folder, 'config.sav'))
    return load_model(os.path.join(model_folder, 'model.sav'))


# Save your trained model.
def save_challenge_model(model_folder, classes, classifier):
    filename = os.path.join(model_folder, 'model.sav')
    save_model(classifier, filename)
    save_config(os.path.join(model_folder, 'config.sav'))

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
<<<<<<< HEAD
=======

def to_one_hot_bool(data, num_classes):
    """
    convert a numpy array to one hot bool
    data: [N] the array that needs to be converted
    num_classes: how many classes in this array
    """
    result = np.eye(num_classes)[data].astype(np.bool_)
    return result

def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def download_model_from_url(url, model_dir, file_name):
    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
>>>>>>> ad44dd9 (change to download from url)
