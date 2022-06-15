#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from soundfile import SoundFile
from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from urllib.request import urlopen, Request
from urllib.parse import urlparse 
import urllib.request
import tensorflow as tf

from tqdm import tqdm
from zipfile import ZipFile
from loguru import logger
import glob
from autokeras.keras_layers import CastToFloat32
from sklearn.preprocessing import OneHotEncoder
from uuid import uuid4

################################################################################
#
# Audio preprocessing
#
################################################################################

import librosa
import IPython.display as ipd
from scipy.signal import butter, lfilter
import os
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import sys
import glob
import librosa.display
import math

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from tqdm.notebook import tqdm
import json

hop_length = 256
REAL_SR = 4000
FIGURE_SIZE = 3

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def clean_frequency_heartbeat(wavfile_path):
  x, sr_fake = librosa.load(wavfile_path, sr=REAL_SR)
  sr = REAL_SR
  min_f = 20
  max_f = 786 #Reference: https://bmcpediatr.biomedcentral.com/track/pdf/10.1186/1471-2431-7-23.pdf
  y = butter_bandpass_filter(x, min_f, max_f, REAL_SR, order=3)
  y = librosa.util.normalize(y)
  sf.write(wavfile_path[:-len(".wav")] + "_clean.wav", y, REAL_SR, 'PCM_16')

def increase_volume(path, level=3):
  song = AudioSegment.from_wav(path)
  # boost volume by 6dB
  louder_song = song + level
  	#save louder song 
  louder_song.export(path, format='wav')

def generate_mel_image(x, sr, hop_length=128):
  total_duration = len(x)/sr 
  fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(FIGURE_SIZE,FIGURE_SIZE))
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
  M = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=256, hop_length=hop_length)
  cmap = 'gist_ncar'
  # cmap = "nipy_spectral"
  sns.heatmap(pd.DataFrame(librosa.power_to_db(M, ref=np.max))[::-1], cmap=cmap, ax=ax, cbar=False)
  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
              hspace = 0, wspace = 0)
  plt.margins(0,0)
  # ax.set_xlim([2, 4])
  ax.set_frame_on(False)
  fig.canvas.draw()
  temp_canvas = fig.canvas
  fig.clf()
  plt.close()
  pil_image = PIL.Image.frombytes('RGB', temp_canvas.get_width_height(),  temp_canvas.tostring_rgb())
  return pil_image

def generate_mel_wav_crops(filepath, output_folder, seconds_window=2):
  # print(filepath)
  x, sr_fake = librosa.load(filepath, sr=REAL_SR)
  sr = REAL_SR
  duration_seconds = len(x)/sr
  for end_time in range(seconds_window, math.floor(duration_seconds), seconds_window):
    # Crop sound
    start_time = end_time - seconds_window
    x_cut = x[start_time*sr: start_time*sr + seconds_window*sr]
    output_filepath_prefix = output_folder.strip("/") + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + "_{}_to_{}".format(int(start_time), int(end_time)) 
    output_filepath_wav =  output_filepath_prefix + ".wav" 
    output_filepath_image = output_filepath_prefix + ".png" 
    # librosa.output.write_wav(filepath, x_cut, sr, norm=True)
    # x_cut = librosa.util.normalize(x_cut)
    pil_image = generate_mel_image(x_cut, sr, hop_length=32)
    pil_image.save(output_filepath_image)
    # sf.write(output_filepath_wav, x_cut, sr, 'PCM_16')
    pil_image.close()
    del x_cut
    del pil_image
  del x
  del sr
  # librosa.cache.clear()

def generate_data_json_file(folder):
  wavfiles = glob.glob(folder.strip("/") + os.path.sep + "*.wav")
  datafiles = []
  url_base = "http://algodev.matheusaraujo.com:8888/training_data/noise_annotations/"
  for wavfile in wavfiles:
    pngfile = os.path.sep.join(os.path.splitext(wavfile)[:-1]) + ".png"
    datafiles.append({"data": {
        "audio" : url_base + os.path.basename(wavfile),
        "image" : url_base + os.path.basename(pngfile)
    }})
  output_pointer = open("datafiles.json", "w")
  json.dump(datafiles, output_pointer)
  output_pointer.close()

def clean_frequency_folder(folder):
  to_clean = glob.glob(folder + "/*_clean.wav")
  for filepath in to_clean:
    os.remove(filepath)
  wav_files = glob.glob(folder + "/*.wav")
  for wav_file in tqdm(wav_files):
    clean_frequency_heartbeat(wav_file)

if __name__=="__main__":
  folder = sys.argv[1]
  wav_files = glob.glob(folder + "*.wav")
  for wav_file in wav_files:
    clean_frequency_heartbeat(wav_file)
    increase_volume(wav_file)

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
    keras_model_folder_to_load = glob.glob(destiny_model + os.path.sep + "*")[0]
    return load_model(keras_model_folder_to_load, custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
    

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    
    if LOAD_TRAINED_MODELS:
        noise_model = load_pretrained_model(model_folder, "noise")
        logger.info(noise_model.summary())
        murmur_model = load_pretrained_model(model_folder, "murmur")
        logger.info(murmur_model.summary())
        murmur_decision_model = load_pretrained_model(model_folder, "murmur_decision")
        logger.info(murmur_decision_model.summary())
    
    # import ipdb;ipdb.set_trace()
    # pass
    
    # # Find data files.
    # if verbose >= 1:
    #     print('Finding data files...')

    # # Find the patient data files.
    # patient_files = find_patient_files(data_folder)
    # num_patient_files = len(patient_files)

    # if num_patient_files==0:
    #     raise Exception('No data was provided.')

    

    # # Extract the features and labels.
    # if verbose >= 1:
    #     print('Extracting features and labels from the Challenge data...')

    # murmur_classes = ['Present', 'Unknown', 'Absent']
    # num_murmur_classes = len(murmur_classes)
    # outcome_classes = ['Abnormal', 'Normal']
    # num_outcome_classes = len(outcome_classes)

    # features = list()
    # murmurs = list()
    # outcomes = list()

    # for i in range(num_patient_files):
    #     if verbose >= 2:
    #         print('    {}/{}...'.format(i+1, num_patient_files))

    #     # Load the current patient data and recordings.
    #     current_patient_data = load_patient_data(patient_files[i])
    #     current_recordings = load_recordings(data_folder, current_patient_data)

    #     # Extract features.
    #     current_features = get_features(current_patient_data, current_recordings)
    #     features.append(current_features)

    #     # Extract labels and use one-hot encoding.
    #     current_murmur = np.zeros(num_murmur_classes, dtype=int)
    #     murmur = get_murmur(current_patient_data)
    #     if murmur in murmur_classes:
    #         j = murmur_classes.index(murmur)
    #         current_murmur[j] = 1
    #     murmurs.append(current_murmur)

    #     current_outcome = np.zeros(num_outcome_classes, dtype=int)
    #     outcome = get_outcome(current_patient_data)
    #     if outcome in outcome_classes:
    #         j = outcome_classes.index(outcome)
    #         current_outcome[j] = 1
    #     outcomes.append(current_outcome)

    # features = np.vstack(features)
    # murmurs = np.vstack(murmurs)
    # outcomes = np.vstack(outcomes)

    # # Train the model.
    # if verbose >= 1:
    #     print('Training model...')

    # # Define parameters for random forest classifier.
    # n_estimators   = 123  # Number of trees in the forest.
    # max_leaf_nodes = 45   # Maximum number of leaf nodes in each tree.
    # random_state   = 6789 # Random state; set for reproducibility.

    # imputer = SimpleImputer().fit(features)
    # features = imputer.transform(features)
    # murmur_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, murmurs)
    # outcome_classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes)

    # # Save the model.
    save_challenge_model(model_folder, noise_model, murmur_model, murmur_decision_model)

    if verbose >= 1:
        print('Done.')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    noise_model = load_model(os.path.join(model_folder, 'noise_model.h5'), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
    murmur_model = load_model(os.path.join(model_folder, 'murmur_model.h5'), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
    murmur_decision_model = load_model(os.path.join(model_folder, 'murmur_decision_model.h5'), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
    return {"noise_model" : noise_model, "murmur_model" : murmur_model, "murmur_decision_model": murmur_decision_model}

def get_unique_name():
    return str(uuid4())

AUX_FOLDER = "recordings_aux"
AUX_IMGS_FOLDER = "images_aux"
def process_recordings(recording):
    recording_name = get_unique_name()
    
# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    os.makedirs(AUX_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_FOLDER, exist_ok=True)
    
    if os.path.exists(AUX_FOLDER):
        files = glob.glob(os.path.join(AUX_FOLDER, "*"))
        for filepath in files:
            os.remove(filepath)
    
    if os.path.exists(AUX_IMGS_FOLDER):
        files = glob.glob(os.path.join(AUX_IMGS_FOLDER, "*"))
        for filepath in files:
            os.remove(filepath)
    
    for recording in recordings:
        recording = recording / 2 ** (16 - 1) # Normalize as: https://stackoverflow.com/questions/50062358/difference-between-load-of-librosa-and-read-of-scipy-io-wavfile
        recording_name = get_unique_name()
        filename =  os.path.join(AUX_FOLDER, recording_name) + ".wav"
        with SoundFile(filename, 'w', REAL_SR, 1, 'PCM_16') as f:
            f.write(np.array(recording).astype(float))
            f.flush()
            f.close()
        logger.info("File saved: {}".format(filename))
    clean_frequency_folder(AUX_FOLDER)
    clean_files = glob.glob(os.path.join(AUX_FOLDER, "*_clean.wav"))
    for clean_file in clean_files:
        generate_mel_wav_crops(clean_file, AUX_IMGS_FOLDER)

    imgs_to_filter = tf.keras.utils.image_dataset_from_directory(AUX_IMGS_FOLDER, labels = None, image_size=(108, 108))
    imgs_noise_prediction = model["noise_model"].predict(imgs_to_filter)
    imgs_noise_df = pd.DataFrame({"imgs_path":imgs_to_filter.file_paths, "noise_prob" : imgs_noise_prediction.flatten()})
    imgs_clean = imgs_noise_df[imgs_noise_df["noise_prob"] < 0.5]
    imgs_noisy = imgs_noise_df[~imgs_noise_df["imgs_path"].isin(imgs_clean["imgs_path"])]
    if imgs_clean.shape[0] == 0:
        logger.info("No clean sound. By default abnormal.")
        labels = [1,0,0] + [1, 0]
        probabilities = [1,0,0] + [1, 0]
        classes = murmur_classes + outcome_classes
        return classes, labels, probabilities
    
    # Delete noisy imgs    
    imgs_noisy["imgs_path"].apply(lambda x: os.remove(x))
    
    # Get murmur embeddings
    murmur_embeddings_model = tf.keras.models.Sequential(model["murmur_model"].layers[:-2])
    imgs_to_emb = tf.keras.utils.image_dataset_from_directory(AUX_IMGS_FOLDER, labels = None, image_size=(72, 72))
    predictions = murmur_embeddings_model.predict(imgs_to_emb)
    embs_df = pd.DataFrame(predictions).sample(frac=1, random_state=42)
    SAMPLE_NUMBERS = 15
    if embs_df.shape[0] < SAMPLE_NUMBERS:
        embs_df = embs_df.sample(SAMPLE_NUMBERS, replace=True, random_state=42)
    else:
        embs_df = embs_df.head(SAMPLE_NUMBERS)
    
    prediction = model["murmur_decision_model"].predict(embs_df.values.flatten().reshape(1,-1))
    present_prob = prediction[0][0]
    absent_prob = 1 - present_prob
    
    murmur_probabilities = np.array([absent_prob, 0, present_prob])
    outcome_probabilities = np.array([absent_prob, present_prob])
    probabilities = np.concatenate([murmur_probabilities, outcome_probabilities])
    
    murmur_labels = murmur_probabilities > 0.5
    outcome_labels = outcome_probabilities > 0.5
    labels = np.concatenate([murmur_labels, outcome_labels])
    classes = murmur_classes + outcome_classes
    # import ipdb;ipdb.set_trace()
    # pass

    # imputer = model['imputer']
    # murmur_classes = model['murmur_classes']
    # murmur_classifier = model['murmur_classifier']
    # outcome_classes = model['outcome_classes']
    # outcome_classifier = model['outcome_classifier']

    # # Load features.
    # features = get_features(data, recordings)

    # # Impute missing data.
    # features = features.reshape(1, -1)
    # features = imputer.transform(features)

    # # Get classifier probabilities.
    # murmur_probabilities = murmur_classifier.predict_proba(features)
    # murmur_probabilities = np.asarray(murmur_probabilities, dtype=np.float32)[:, 0, 1]
    # outcome_probabilities = outcome_classifier.predict_proba(features)
    # outcome_probabilities = np.asarray(outcome_probabilities, dtype=np.float32)[:, 0, 1]

    # # Choose label with highest probability.
    # murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    # idx = np.argmax(murmur_probabilities)
    # murmur_labels[idx] = 1
    # outcome_labels = np.zeros(len(outcome_classes), dtype=np.int_)
    # idx = np.argmax(outcome_probabilities)
    # outcome_labels[idx] = 1

    # # Concatenate classes, labels, and probabilities.
    # classes = murmur_classes + outcome_classes
    # labels = np.concatenate((murmur_labels, outcome_labels))
    # probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, noise_model, murmur_model, murmur_decision_model):
    noise_model.save(os.path.join(model_folder, "noise_model.h5"))
    murmur_model.save(os.path.join(model_folder, "murmur_model.h5"))
    murmur_decision_model.save(os.path.join(model_folder, "murmur_decision_model.h5"))
    

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