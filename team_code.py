#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from gc import callbacks
from pyclbr import Function
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
from tensorflow.python.keras.utils.tf_utils import ListWrapper
from tqdm import tqdm
from zipfile import ZipFile
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from loguru import logger
from numpy import *
import glob
from autokeras.keras_layers import CastToFloat32
from sklearn.preprocessing import OneHotEncoder
from uuid import uuid4
import shutil
import tensorflow_datasets as tfds
from sklearn.preprocessing import OneHotEncoder

tf.keras.utils.set_random_seed(42)
tf.config.experimental_run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

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
from tqdm import tqdm
import json
from keras.engine.functional import Functional
import tensorflow_addons as tfa

hop_length = 256
REAL_SR = 4000
FIGURE_SIZE = 3
EMBDS_PER_PATIENTS = 15
train_folder_murmur = "train_folder_murmur"
test_folder_murmur = "test_folder_murmur"
val_folder_murmur = "val_folder_murmur"

val_positive_folder = val_folder_murmur + os.path.sep + "positive"
val_negative_folder = val_folder_murmur + os.path.sep + "negative"
test_positive_folder = test_folder_murmur + os.path.sep + "positive"
test_negative_folder = test_folder_murmur + os.path.sep + "negative"
train_positive_folder = train_folder_murmur + os.path.sep + "positive"
train_negative_folder = train_folder_murmur + os.path.sep + "negative"

train_outcome_folder = "train_outcome_folder"
test_outcome_folder = "test_outcome_folder"
val_outcome_folder = "val_outcome_folder"

val_outcome_positive_folder = val_outcome_folder + os.path.sep + "positive"
val_outcome_negative_folder = val_outcome_folder + os.path.sep + "negative"
test_outcome_positive_folder = test_outcome_folder + os.path.sep + "positive"
test_outcome_negative_folder = test_outcome_folder + os.path.sep + "negative"
train_outcome_positive_folder = train_outcome_folder + os.path.sep + "positive"
train_outcome_negative_folder = train_outcome_folder + os.path.sep + "negative"

train_embs_folder_murmur = "train_embs_folder_murmur"
test_embs_folder_murmur = "test_embs_folder_murmur"
val_embs_folder_murmur = "val_embs_folder_murmur"

val_embs_positive_folder = val_embs_folder_murmur + os.path.sep + "positive"
val_embs_negative_folder = val_embs_folder_murmur + os.path.sep + "negative"
test_embs_positive_folder = test_embs_folder_murmur + os.path.sep + "positive"
test_embs_negative_folder = test_embs_folder_murmur + os.path.sep + "negative"
train_embs_positive_folder = train_embs_folder_murmur + os.path.sep + "positive"
train_embs_negative_folder = train_embs_folder_murmur + os.path.sep + "negative"


LOAD_TRAINED_MODELS = True
TRAIN_NOISE_DETECTION = False

NOISE_IMAGE_SIZE = (64, 64)
MURMUR_IMAGE_SIZE = NOISE_IMAGE_SIZE
EMBS_SIZE = 64



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
    pil_image = generate_mel_image(x_cut, sr, hop_length=32)
    output_filepath_prefix = output_folder.strip("/") + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + "_{}_to_{}".format(int(start_time), int(end_time)) 
    output_filepath_wav =  output_filepath_prefix + ".wav" 
    output_filepath_image = output_filepath_prefix + ".png" 
    # librosa.output.write_wav(filepath, x_cut, sr, norm=True)
    # x_cut = librosa.util.normalize(x_cut)
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
    
def clean_folder(folder_path):
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, "*"))
        for filepath in files:
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except:
                logger.info("Could not delete: {}".format(filepath))
                
NOISE_DETECTION_WORKING_DIR = "/dev/shm" 
NOISE_DETECTION_IMGS_PATH = os.path.join(NOISE_DETECTION_WORKING_DIR, "noise_detection_sandbox")

def get_all_metrics():
    AUC_metric = tf.keras.metrics.AUC(name="auc", curve="PR")
    AUC_ROC_metric = tf.keras.metrics.AUC(name="auc_roc", curve="ROC")
    FN_metric = tf.keras.metrics.FalseNegatives()
    FP_metric = tf.keras.metrics.FalsePositives()
    ACC_metric = tf.keras.metrics.Accuracy()
    PRECISION_metric = tf.keras.metrics.Precision()
    RECALL_metric = tf.keras.metrics.Recall()
    WEIGHTED_ACC = compute_weighted_accuracy
    all_metrics = [AUC_metric, FN_metric, FP_metric, ACC_metric, PRECISION_metric, RECALL_metric, AUC_ROC_metric, WEIGHTED_ACC]
    return all_metrics

# Load recordings.
def load_recordings_custom(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    locations = list()

    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)
        locations.append(entries[0].upper())
        

    if get_frequencies:
        return recordings, frequencies, locations
    else:
        return recordings, locations

def get_murmur_locations(data):
    for line in data.split('\n'):
        if line.startswith('#Murmur locations: '):
            try:
                locations = line.upper().split(": ")[1].strip().split('+')
                if locations[0] == "NAN":
                    return np.nan
                else:
                    return locations
            except:
                pass
    raise Exception("No murmur location found.")

def clean_current_path():
    shutil.rmtree("recordings_aux", ignore_errors=True)
    shutil.rmtree("image_aux", ignore_errors=True)
    shutil.rmtree("train_murmur_folder", ignore_errors=True)
    shutil.rmtree("test_murmur_folder", ignore_errors=True)
    shutil.rmtree("val_murmur_folder", ignore_errors=True)
    shutil.rmtree("train_outcome_folder", ignore_errors=True)
    shutil.rmtree("test_outcome_folder", ignore_errors=True)
    shutil.rmtree("val_outcome_folder", ignore_errors=True)
    shutil.rmtree("train_embs_folder_murmur", ignore_errors=True)
    shutil.rmtree("test_embs_folder_murmur", ignore_errors=True)
    shutil.rmtree("val_embs_folder_murmur", ignore_errors=True)

def prepare_train_val_test_murmur_folders():    
    shutil.rmtree("train_murmur_folder", ignore_errors=True)
    shutil.rmtree("test_murmur_folder", ignore_errors=True)
    shutil.rmtree("val_murmur_folder", ignore_errors=True)
    shutil.rmtree("train_outcome_folder", ignore_errors=True)
    shutil.rmtree("test_outcome_folder", ignore_errors=True)
    shutil.rmtree("val_outcome_folder", ignore_errors=True)
    shutil.rmtree("train_embs_folder_murmur", ignore_errors=True)
    shutil.rmtree("test_embs_folder_murmur", ignore_errors=True)
    shutil.rmtree("val_embs_folder_murmur", ignore_errors=True)

    if not os.path.exists(train_folder_murmur):
        os.mkdir(train_folder_murmur)
        os.mkdir(train_positive_folder)
        os.mkdir(train_negative_folder)

    if not os.path.exists(val_folder_murmur):
        os.mkdir(val_folder_murmur)
        os.mkdir(val_positive_folder)
        os.mkdir(val_negative_folder)

    if not os.path.exists(test_folder_murmur):
        os.mkdir(test_folder_murmur)
        os.mkdir(test_positive_folder)
        os.mkdir(test_negative_folder)

    if not os.path.exists(train_outcome_folder):
        os.mkdir(train_outcome_folder)
        os.mkdir(train_outcome_positive_folder)
        os.mkdir(train_outcome_negative_folder)

    if not os.path.exists(val_outcome_folder):
        os.mkdir(val_outcome_folder)
        os.mkdir(val_outcome_positive_folder)
        os.mkdir(val_outcome_negative_folder)

    if not os.path.exists(test_outcome_folder):
        os.mkdir(test_outcome_folder)
        os.mkdir(test_outcome_positive_folder)
        os.mkdir(test_outcome_negative_folder)
    
    if not os.path.exists(train_embs_folder_murmur):
        os.mkdir(train_embs_folder_murmur)
        os.mkdir(train_embs_positive_folder)
        os.mkdir(train_embs_negative_folder)
    
    if not os.path.exists(val_embs_folder_murmur):
        os.mkdir(val_embs_folder_murmur)
        os.mkdir(val_embs_positive_folder)
        os.mkdir(val_embs_negative_folder)
    
    if not os.path.exists(test_embs_folder_murmur):
        os.mkdir(test_embs_folder_murmur)
        os.mkdir(test_embs_positive_folder)
        os.mkdir(test_embs_negative_folder)


def load_image_array(filepath):
  image = tf.keras.preprocessing.image.load_img(filepath, target_size=MURMUR_IMAGE_SIZE)
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  return input_arr

def generate_patient_embeddings_folder(row_patient, murmur_model):
    patient_id = row_patient["patient"]
    split = row_patient["split"]
    label = row_patient["label"]
    destiny_folder = split + "_embs_folder_murmur" + os.path.sep +  label
    original_files_query = split + "_folder_murmur" + os.path.sep + label + os.path.sep + patient_id + "*"
    patient_files = glob.glob(original_files_query)
    embs = []
    for patient_file in patient_files:
        img_array = load_image_array(patient_file)
        prediction = murmur_model.predict(img_array, verbose=0)
        embs.append(prediction[0])
        
    embs = np.array(embs)
    embs_df = pd.DataFrame(embs).sample(frac=1, random_state=42)
    # We lose potential embds here, fix later
    if embs_df.shape[0] < EMBDS_PER_PATIENTS:
        embs_df = embs_df.sample(EMBDS_PER_PATIENTS, replace=True, random_state=42)
    else:
        embs_df = embs_df.head(EMBDS_PER_PATIENTS)
    destiny_file = destiny_folder + os.path.sep + patient_id
    embs_df.to_pickle(destiny_file)
    # logger.info("Saved: " + destiny_file)
    

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

def load_embs_labels(folder, problem, patient_murmur_outcome_df):
  all_files = glob.glob(folder + os.path.sep + "**/*")
  embs = []
  labels = []
  for patient_file in all_files:
    patient_df = pd.read_pickle(patient_file)
    patient_id = patient_file.split("/")[-1]
    label = patient_murmur_outcome_df[patient_murmur_outcome_df["patient_id"] == patient_id].iloc[0][problem]
    label = True if label == "Present" or label == "Abnormal" else False
    embs.append(patient_df.values.flatten())
    labels.append(label)
  return embs, labels

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    AUX_FOLDER = "recordings_aux"
    AUX_IMGS_FOLDER = "images_aux"
    AUX_IMGS_POSITIVE_FOLDER = os.path.join(AUX_IMGS_FOLDER, "positive")
    AUX_IMGS_NEGATIVE_FOLDER = os.path.join(AUX_IMGS_FOLDER, "negative")
    murmur_image_folders = [train_positive_folder, train_negative_folder, val_positive_folder, val_negative_folder, test_positive_folder, test_negative_folder]
    # clean_current_path()
    
    os.makedirs(AUX_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_POSITIVE_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_NEGATIVE_FOLDER, exist_ok=True)
    
    
    
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    os.system("tar -xf noise_detection_sandbox.tar.gz -C {}".format(NOISE_DETECTION_WORKING_DIR))

    if LOAD_TRAINED_MODELS:
        # noise_model = load_pretrained_model(model_folder, "noise")
        logger.info("Loading model: {}".format("noises"))
        noise_model = load_model(os.path.join(model_folder, "noise_model.tf"))
        logger.info(noise_model.summary())
        murmur_model = load_pretrained_model(model_folder, "murmur")
        logger.info(murmur_model.summary())
        murmur_decision_model = load_pretrained_model(model_folder, "murmur_decision")
        logger.info(murmur_decision_model.summary())
        save_challenge_model(model_folder, noise_model, murmur_model, murmur_decision_model)
        if verbose >= 1:
            print('Done.')

    else:
        pass
    # Noise model - Parameters found after runnign AutoKeras
    if TRAIN_NOISE_DETECTION:
        # noise_input_config, noise_cast_to_float_config, noise_normalization_config, noise_xception_config, noise_global_average_config, noise_dense_config, noise_classification_head_config = get_noise_model_configs()
        
        # noise_layer_0 = tf.keras.Input(**noise_input_config)
        # noise_layer_1 = CastToFloat32.from_config(noise_cast_to_float_config)
        # noise_layer_2 = tf.keras.layers.Normalization.from_config(noise_normalization_config)
        # # noise_layer_3 = Functional.from_config(noise_xception_config)
        # noise_layer_3a = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        # noise_layer_3b = tf.keras.layers.MaxPooling2D((2, 2))
        # noise_layer_3c = tf.keras.layers.Conv2D(8, (3, 3), activation='relu')
        
        # noise_layer_4 = tf.keras.layers.GlobalAveragePooling2D.from_config(noise_global_average_config)
        # noise_layer_5 = tf.keras.layers.Dense.from_config(noise_dense_config)
        # noise_layer_6 = tf.keras.layers.Activation.from_config(noise_classification_head_config)
        # noise_model_new = tf.keras.Sequential([noise_layer_0, noise_layer_1, noise_layer_2, tf.keras.applications.xception.Xception(
        # include_top=False,
        # input_tensor=None,
        # weights=None,
        # pooling=None,
        # classes=2,
        # classifier_activation='softmax'
        # ), 
        # noise_layer_4,
        # noise_layer_5, noise_layer_6])
        # assert np.sum([np.prod(v.get_shape().as_list()) for v in noise_model.trainable_variables]) == np.sum([np.prod(v.get_shape().as_list()) for v in noise_model_new.trainable_variables]), "Sanity check, leave commented"
        # optimizer = tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.0001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False})
        # noise_model_new.compile(optimizer=optimizer, metrics=get_all_metrics(), loss="binary_crossentropy",)
        
        noise_model_new = tf.keras.models.Sequential()
        noise_model_new.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(NOISE_IMAGE_SIZE[0], NOISE_IMAGE_SIZE[1], 3)))
        noise_model_new.add(tf.keras.layers.MaxPooling2D((2, 2)))
        noise_model_new.add(tf.keras.layers.Dropout(.2))
        noise_model_new.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(NOISE_IMAGE_SIZE[0], NOISE_IMAGE_SIZE[1], 3)))
        noise_model_new.add(tf.keras.layers.MaxPooling2D((2, 2)))
        noise_model_new.add(tf.keras.layers.Dropout(.2))
        noise_model_new.add(tf.keras.layers.Flatten())
        noise_model_new.add(tf.keras.layers.Dropout(.5))
        noise_model_new.add(tf.keras.layers.Dense(EMBS_SIZE, activation='relu'))
        noise_model_new.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        noise_model_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.0001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), 
                loss="binary_crossentropy",
                metrics=get_all_metrics())
        batch_size = 1
        
        noise_detection_dataset_train_val = tf.keras.utils.image_dataset_from_directory(NOISE_DETECTION_IMGS_PATH, subset="training", validation_split=0.2, batch_size=batch_size, seed=42, image_size=MURMUR_IMAGE_SIZE, )
        dataset_size = len(noise_detection_dataset_train_val)
        noise_detection_dataset_train_val = noise_detection_dataset_train_val.shuffle(buffer_size=dataset_size)
        noise_detection_dataset_train = noise_detection_dataset_train_val.take(int(dataset_size * 0.7))
        noise_detection_dataset_val = noise_detection_dataset_train_val.skip(int(dataset_size * 0.7))
        noise_detection_dataset_test = tf.keras.utils.image_dataset_from_directory(NOISE_DETECTION_IMGS_PATH, subset="validation", validation_split=0.2, batch_size=batch_size, seed=42, image_size=MURMUR_IMAGE_SIZE, )
        

        noise_model_new.fit(noise_detection_dataset_train, epochs = 100, callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0,
            patience=20,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True,
        )], validation_data=noise_detection_dataset_val)

        logger.info("Noise Model Classification Report")
        logger.info(noise_model_new.evaluate(noise_detection_dataset_test, return_dict=True))
        tf.keras.models.save_model(
            noise_model_new,
            os.path.join(model_folder, 'noise_model.tf'),
            overwrite=True,
            include_optimizer=True,
            save_format="tf",
            signatures=None,
            options=None,
            save_traces=True
        )
    else:
        noise_model_new = noise_model
    
    
    # ROOT_FOLDER = "/dev/shm/noise_imgs"
    # ROOT_IMAGES = "/content/output/"
    
    # HAS_NOISE_FOLDER = ROOT_FOLDER + "has_noise/"
    # HAS_HEARTBEAT_FOLDER = ROOT_FOLDER + "has_heatbeat/"

    # if not os.path.exists(ROOT_FOLDER):
    #     os.mkdir(ROOT_FOLDER)
    # if not os.path.exists(HAS_NOISE_FOLDER):
    #     os.mkdir(HAS_NOISE_FOLDER)
    # else:
    #     shutil.rmtree(HAS_NOISE_FOLDER)
    #     os.mkdir(HAS_NOISE_FOLDER)
    # if not os.path.exists(HAS_HEARTBEAT_FOLDER):
    #     os.mkdir(HAS_HEARTBEAT_FOLDER)
    # else:
    #     shutil.rmtree(HAS_HEARTBEAT_FOLDER)
    #     os.mkdir(HAS_HEARTBEAT_FOLDER)
        
    # for row_index, row in tqdm(labels_df.iterrows()):
    #     image_path = ROOT_IMAGES + row["image"].split("/")[-1]
    #     if row["has_noise"]:
    #         shutil.copy2(image_path, HAS_NOISE_FOLDER)
    #     else:
    #         shutil.copy2(image_path, HAS_HEARTBEAT_FOLDER)
    
    
        
    # import ipdb;ipdb.set_trace()
    # pass
    
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    

    # # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    murmurs = list()
    outcomes = list()
    
    patients_file_informations = []
    patient_ids = []
    patients_murmurs_labels = []
    patients_outcomes_labels = []
    for i in tqdm(range(num_patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        patient_id = get_patient_id(current_patient_data)
        murmur_label = get_murmur(current_patient_data)
        outcome_label = get_outcome(current_patient_data)
        patient_ids.append(patient_id)
        patients_murmurs_labels.append(murmur_label)
        patients_outcomes_labels.append(outcome_label)
    
    patient_murmur_outcome_df = pd.DataFrame({
        "patient_id" : patient_ids,
        "murmur" : patients_murmurs_labels,
        "outcome" : patients_outcomes_labels
    })
    
    unique_original_files = pd.Series(pd.Series(patient_ids).unique())
    test_set = unique_original_files.sample(frac=0.2, random_state=42)
    train_val_set = unique_original_files[~unique_original_files.isin(test_set)]
    train_set = train_val_set.sample(frac=0.8, random_state=42)
    val_set = train_val_set[~train_val_set.isin(train_set)]
    
    for i in tqdm(range(num_patient_files)):
    #     # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings, locations = load_recordings_custom(data_folder, current_patient_data)
        murmur_locations = get_murmur_locations(current_patient_data)
        patient_id = get_patient_id(current_patient_data)
        
        for recording, location in zip(current_recordings, locations):
            recording = recording / 2 ** (16 - 1) # Normalize as: https://stackoverflow.com/questions/50062358/difference-between-load-of-librosa-and-read-of-scipy-io-wavfile
            recording_name = str(patient_id) + "_" + get_unique_name()
            patients_file_informations.append({
                "file_prefix" : recording_name,
                "murmur" : get_murmur(current_patient_data),
                "outcome" : get_outcome(current_patient_data),
                "murmur_locations" : murmur_locations,
                "location" : location,
                "patient_id" : patient_id
            })
            filename = os.path.join(AUX_FOLDER, recording_name) + ".wav"
            with SoundFile(filename, 'w', REAL_SR, 1, 'PCM_16') as f:
                f.write(np.array(recording).astype(float))
                f.flush()
                f.close()
            # logger.info("File saved: {}".format(filename))
    patients_file_informations_df = pd.DataFrame(patients_file_informations)
    patients_file_informations_df["is_murmur_location"] = patients_file_informations_df.apply(lambda row: type(row["murmur_locations"]) == list and row["location"] in row["murmur_locations"], axis=1)
    
    clean_frequency_folder(AUX_FOLDER)
    
    clean_files = glob.glob(os.path.join(AUX_FOLDER, "*_clean.wav"))
    logger.info("Generating melspectograms crops")
    prepare_train_val_test_murmur_folders()
    for clean_file in tqdm(clean_files): 
        file_prefix = os.path.basename(clean_file).split("_clean")[0]
        file_info = patients_file_informations_df[patients_file_informations_df['file_prefix'] == file_prefix].iloc[0]
        if file_info["murmur"] == "Unknown":
            continue
        if file_info["murmur"] == "Present" and file_info["is_murmur_location"]:
            if file_info["patient_id"] in train_set.values:
                destiny_folder = train_positive_folder
            if file_info["patient_id"] in val_set.values:
                destiny_folder = val_positive_folder
            if file_info["patient_id"] in test_set.values:
                destiny_folder = test_positive_folder
        if file_info["murmur"] == "Absent":
            if file_info["patient_id"] in train_set.values:
                destiny_folder = train_negative_folder
            if file_info["patient_id"] in val_set.values:
                destiny_folder = val_negative_folder
            if file_info["patient_id"] in test_set.values:
                destiny_folder = test_negative_folder             
        generate_mel_wav_crops(clean_file, destiny_folder)

    # Clean imgs with noise prediction
    
    noise_murmur_df_list = []
    for folder in murmur_image_folders:
        try:
            noise_detection_dataset = tf.keras.utils.image_dataset_from_directory(folder, label_mode = None, image_size=NOISE_IMAGE_SIZE, shuffle=False)
        except ValueError:
            continue
        predictions = noise_model.predict(noise_detection_dataset)
        noise_murmur_df_list.append(pd.DataFrame({"predictions" : predictions.flatten(), "filepath" : noise_detection_dataset.file_paths}))
    noise_murmur_df = pd.concat(noise_murmur_df_list)        

    files_to_exclude = noise_murmur_df[noise_murmur_df["predictions"] > 0.5]
    logger.info("Remove noisy files")
    if files_to_exclude.shape[0] > 0:
        for filepath in tqdm(files_to_exclude["filepath"]):
            os.remove(filepath)
    
    
    
    # Dataset for final decision
    all_files = []
    for folder in murmur_image_folders:
        all_files = all_files + glob.glob("{}/*".format(folder))
    
    patient_ids = pd.Series(all_files).str.split("/").str[2].str.split("_").str[0]
    split = pd.Series(all_files).str.split("/").str[0].str.split("_").str[0]
    label = pd.Series(all_files).str.split("/").str[1]
    patient_split_df = pd.concat([patient_ids, split, label],axis=1)
    patient_split_df.columns = ["patient", "split", "label"]
    patient_split_df = patient_split_df.drop_duplicates(subset=["patient"])
    
    # Load dataset for murmur model training
    class_weight = {0: 1, 1: 1.5}  
    batch_size = 16
    murmur_detection_dataset_train = tf.keras.utils.image_dataset_from_directory(train_folder_murmur, batch_size=batch_size, seed=42, image_size=NOISE_IMAGE_SIZE )
    murmur_detection_dataset_val = tf.keras.utils.image_dataset_from_directory(val_folder_murmur, batch_size=batch_size, seed=42, image_size=NOISE_IMAGE_SIZE )
    murmur_detection_dataset_test = tf.keras.utils.image_dataset_from_directory(test_folder_murmur, batch_size=batch_size, seed=42, image_size=NOISE_IMAGE_SIZE, )
    
    # murmur_config = get_murmur_model_configs()
    # murmur_model_new = Functional.from_config(murmur_config) 
    # optimizer = tfa.optimizers.AdamW(
    #             learning_rate=0.0001,
    #             weight_decay=0.01,
    #             beta_1=0.9,
    #             beta_2=0.999,
    #             epsilon=1e-6,
    #             exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    #         )  
    # murmur_model_new.compile(optimizer=optimizer, metrics=get_all_metrics(), loss="binary_crossentropy",)
    murmur_model_new = tf.keras.models.clone_model(noise_model_new)
    murmur_model_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.0001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), loss="binary_crossentropy", metrics=get_all_metrics())
    # murmur_model_new.set_weights(noise_model_new.get_weights())
    murmur_model_new.fit(murmur_detection_dataset_train, validation_data=murmur_detection_dataset_val, epochs = 1, class_weight=class_weight, callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True
        )], workers= os.cpu_count() - 1)
    logger.info("Murmur Model Performance")
    logger.info(murmur_model_new.evaluate(murmur_detection_dataset_test, return_dict=True))
    
    murmur_embedding_model = tf.keras.models.Sequential(murmur_model_new.layers[:-2])
    
    for row_id, row_patient in tqdm(patient_split_df.iterrows(), total=patient_split_df.shape[0]):
        generate_patient_embeddings_folder(row_patient, murmur_embedding_model)
    
    embs_train, labels_train = load_embs_labels(train_embs_folder_murmur, "murmur", patient_murmur_outcome_df)
    embs_val, labels_val = load_embs_labels(val_embs_folder_murmur, "murmur", patient_murmur_outcome_df)
    
    murmur_decision_config = get_murmur_decision_model_configs()
    murmur_decision_new = Functional.from_config(murmur_decision_config) 
    murmur_decision_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.00001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), loss="binary_crossentropy", metrics=get_all_metrics())
    
    
    train_decision_dataset = tf.data.Dataset.from_tensor_slices((embs_train, labels_train))
    val_decision_dataset = tf.data.Dataset.from_tensor_slices((embs_val, labels_val))
    murmur_decision_new.fit(train_decision_dataset, validation_data = val_decision_dataset, epochs = 1, class_weight=class_weight, callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_compute_weighted_accuracy:",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True
        )], workers= os.cpu_count() - 1)
    
    
    # # Save the model.
    save_challenge_model(model_folder, noise_model_new, murmur_model_new, murmur_decision_new)
    # clean_current_path()
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


def process_recordings(recording):
    recording_name = get_unique_name()
    
# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    AUX_FOLDER = "recordings_aux"
    AUX_IMGS_FOLDER = "images_aux"
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    num_murmur_classes = len(murmur_classes)
    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)
    
    os.makedirs(AUX_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_FOLDER, exist_ok=True)
    
    clean_folder(AUX_IMGS_FOLDER)
    clean_folder(AUX_FOLDER)
    
    
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

    imgs_to_filter = tf.keras.utils.image_dataset_from_directory(AUX_IMGS_FOLDER, labels = None, image_size=NOISE_IMAGE_SIZE)
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
    embs = murmur_embeddings_model.predict(imgs_to_emb)
    embs_df = pd.DataFrame(embs).sample(frac=1, random_state=42)

    if embs_df.shape[0] < EMBDS_PER_PATIENTS:
        embs_df = embs_df.sample(EMBDS_PER_PATIENTS, replace=True, random_state=42)
    else:
        embs_df = embs_df.head(EMBDS_PER_PATIENTS)
    
    prediction = model["murmur_decision_model"].predict(embs_df.values.flatten().reshape(1,-1))
    present_prob = prediction[0][0]
    absent_prob = 1 - present_prob
    
    murmur_probabilities = np.array([present_prob, 0, absent_prob])
    outcome_probabilities = np.array([present_prob, absent_prob])
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

def get_murmur_decision_model_configs():
    murmur_decision_config = {'name': 'model', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, EMBS_SIZE * EMBDS_PER_PATIENTS), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}, 'name': 'input_1', 'inbound_nodes': []}, {'class_name': 'Custom>CastToFloat32', 'config': {'name': 'cast_to_float32', 'trainable': True, 'dtype': 'float32'}, 'name': 'cast_to_float32', 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 1024, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense', 'inbound_nodes': [[['cast_to_float32', 0, 0, {}]]]}, {'class_name': 'ReLU', 'config': {'name': 're_lu', 'trainable': True, 'dtype': 'float32', 'max_value': None, 'negative_slope': array(0., dtype=float32), 'threshold': array(0., dtype=float32)}, 'name': 're_lu', 'inbound_nodes': [[['dense', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_1', 'inbound_nodes': [[['re_lu', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'classification_head_1', 'trainable': True, 'dtype': 'float32', 'activation': 'sigmoid'}, 'name': 'classification_head_1', 'inbound_nodes': [[['dense_1', 0, 0, {}]]]}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['classification_head_1', 0, 0]]}
    return murmur_decision_config
def get_murmur_model_configs(): 
    murmur_config = {'name': 'model', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, MURMUR_IMAGE_SIZE[0], MURMUR_IMAGE_SIZE[0], 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}, 'name': 'input_1', 'inbound_nodes': []}, {'class_name': 'Custom>CastToFloat32', 'config': {'name': 'cast_to_float32', 'trainable': True, 'dtype': 'float32'}, 'name': 'cast_to_float32', 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'class_name': 'Resizing', 'config': {'name': 'resizing', 'trainable': True, 'dtype': 'float32', 'height': 224, 'width': 224, 'interpolation': 'bilinear', 'crop_to_aspect_ratio': False}, 'name': 'resizing', 'inbound_nodes': [[['cast_to_float32', 0, 0, {}]]]}, {'class_name': 'Functional', 'config': {'name': 'xception', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 224, 224, 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_2'}, 'name': 'input_2', 'inbound_nodes': []}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv1', 'trainable': True, 'dtype': 'float32', 'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv1', 'inbound_nodes': [[['input_2', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv1_bn', 'trainable': True, 'dtype': 'float32', 'axis':[3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv1_bn', 'inbound_nodes': [[['block1_conv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv1_act', 'inbound_nodes': [[['block1_conv1_bn', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv2', 'trainable': True, 'dtype': 'float32', 'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv2', 'inbound_nodes': [[['block1_conv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv2_bn', 'inbound_nodes': [[['block1_conv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv2_act', 'inbound_nodes': [[['block1_conv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 24}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv1', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv1_bn', 'inbound_nodes': [[['block2_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block2_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block2_sepconv2_act', 'inbound_nodes': [[['block2_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 35}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv2', 'inbound_nodes': [[['block2_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv2_bn', 'inbound_nodes': [[['block2_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block2_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block2_pool', 'inbound_nodes': [[['block2_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization', 'inbound_nodes': [[['conv2d', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add', 'trainable': True, 'dtype': 'float32'}, 'name': 'add', 'inbound_nodes': [[['block2_pool', 0, 0, {}], ['batch_normalization', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv1_act', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 56}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv1', 'inbound_nodes': [[['block3_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv1_bn', 'inbound_nodes': [[['block3_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv2_act', 'inbound_nodes': [[['block3_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 67}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv2', 'inbound_nodes': [[['block3_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv2_bn', 'inbound_nodes': [[['block3_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_1', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block3_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block3_pool', 'inbound_nodes': [[['block3_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_1', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_1', 'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_1', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_1', 'inbound_nodes': [[['block3_pool', 0, 0, {}], ['batch_normalization_1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv1_act', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 88}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv1', 'inbound_nodes': [[['block4_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv1_bn', 'inbound_nodes': [[['block4_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv2_act', 'inbound_nodes': [[['block4_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 99}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv2', 'inbound_nodes': [[['block4_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv2_bn', 'inbound_nodes': [[['block4_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_2', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block4_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block4_pool', 'inbound_nodes': [[['block4_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_2', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_2', 'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_2', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_2', 'inbound_nodes': [[['block4_pool', 0, 0, {}], ['batch_normalization_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv1_act', 'inbound_nodes': [[['add_2', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 120}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv1', 'inbound_nodes': [[['block5_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv1_bn', 'inbound_nodes': [[['block5_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv2_act', 'inbound_nodes': [[['block5_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 131}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv2', 'inbound_nodes': [[['block5_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv2_bn', 'inbound_nodes': [[['block5_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv3_act', 'inbound_nodes': [[['block5_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 142}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv3', 'inbound_nodes': [[['block5_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv3_bn', 'inbound_nodes': [[['block5_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_3', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_3', 'inbound_nodes': [[['block5_sepconv3_bn', 0, 0, {}], ['add_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv1_act', 'inbound_nodes': [[['add_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 154}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv1', 'inbound_nodes': [[['block6_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv1_bn', 'inbound_nodes': [[['block6_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv2_act', 'inbound_nodes': [[['block6_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 165}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv2', 'inbound_nodes': [[['block6_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv2_bn', 'inbound_nodes': [[['block6_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv3_act', 'inbound_nodes': [[['block6_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 176}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv3', 'inbound_nodes': [[['block6_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv3_bn', 'inbound_nodes': [[['block6_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_4', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_4', 'inbound_nodes': [[['block6_sepconv3_bn', 0, 0, {}], ['add_3', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv1_act', 'inbound_nodes': [[['add_4', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 188}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv1', 'inbound_nodes': [[['block7_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv1_bn', 'inbound_nodes': [[['block7_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv2_act', 'inbound_nodes': [[['block7_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 199}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv2', 'inbound_nodes': [[['block7_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv2_bn', 'inbound_nodes': [[['block7_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv3_act', 'inbound_nodes': [[['block7_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 210}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv3', 'inbound_nodes': [[['block7_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv3_bn', 'inbound_nodes': [[['block7_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_5', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_5', 'inbound_nodes': [[['block7_sepconv3_bn', 0, 0, {}], ['add_4', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv1_act', 'inbound_nodes': [[['add_5', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 222}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv1', 'inbound_nodes': [[['block8_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv1_bn', 'inbound_nodes': [[['block8_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv2_act', 'inbound_nodes': [[['block8_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 233}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv2', 'inbound_nodes': [[['block8_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv2_bn', 'inbound_nodes': [[['block8_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv3_act', 'inbound_nodes': [[['block8_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 244}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv3', 'inbound_nodes': [[['block8_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv3_bn', 'inbound_nodes': [[['block8_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_6', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_6', 'inbound_nodes': [[['block8_sepconv3_bn', 0, 0, {}], ['add_5', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv1_act', 'inbound_nodes': [[['add_6', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 256}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv1', 'inbound_nodes': [[['block9_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv1_bn', 'inbound_nodes': [[['block9_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv2_act', 'inbound_nodes': [[['block9_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 267}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv2', 'inbound_nodes': [[['block9_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv2_bn', 'inbound_nodes': [[['block9_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv3_act', 'inbound_nodes': [[['block9_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 278}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv3', 'inbound_nodes': [[['block9_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv3_bn', 'inbound_nodes': [[['block9_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_7', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_7', 'inbound_nodes': [[['block9_sepconv3_bn', 0, 0, {}], ['add_6', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv1_act', 'inbound_nodes': [[['add_7', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 290}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv1', 'inbound_nodes': [[['block10_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv1_bn', 'inbound_nodes': [[['block10_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv2_act', 'inbound_nodes': [[['block10_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 301}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv2', 'inbound_nodes': [[['block10_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv2_bn', 'inbound_nodes': [[['block10_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv3_act', 'inbound_nodes': [[['block10_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 312}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv3', 'inbound_nodes': [[['block10_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv3_bn', 'inbound_nodes': [[['block10_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_8', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_8', 'inbound_nodes': [[['block10_sepconv3_bn', 0, 0, {}], ['add_7', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv1_act', 'inbound_nodes': [[['add_8', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 324}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv1', 'inbound_nodes': [[['block11_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv1_bn', 'inbound_nodes': [[['block11_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv2_act', 'inbound_nodes': [[['block11_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 335}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv2', 'inbound_nodes': [[['block11_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv2_bn', 'inbound_nodes': [[['block11_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv3_act', 'inbound_nodes': [[['block11_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 346}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv3', 'inbound_nodes': [[['block11_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv3_bn', 'inbound_nodes': [[['block11_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_9', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_9', 'inbound_nodes': [[['block11_sepconv3_bn', 0, 0, {}], ['add_8', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv1_act', 'inbound_nodes': [[['add_9', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 358}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv1', 'inbound_nodes': [[['block12_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv1_bn', 'inbound_nodes': [[['block12_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv2_act', 'inbound_nodes': [[['block12_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 369}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv2', 'inbound_nodes': [[['block12_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv2_bn', 'inbound_nodes': [[['block12_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv3_act', 'inbound_nodes': [[['block12_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 380}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv3', 'inbound_nodes': [[['block12_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv3_bn', 'inbound_nodes': [[['block12_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_10', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_10', 'inbound_nodes': [[['block12_sepconv3_bn', 0, 0, {}], ['add_9', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv1_act', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 392}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv1', 'inbound_nodes': [[['block13_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv1_bn', 'inbound_nodes': [[['block13_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv2_act', 'inbound_nodes': [[['block13_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 403}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv2', 'inbound_nodes': [[['block13_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv2_bn', 'inbound_nodes': [[['block13_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_3', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_3', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block13_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block13_pool', 'inbound_nodes': [[['block13_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_3', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_3', 'inbound_nodes': [[['conv2d_3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_11', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_11', 'inbound_nodes': [[['block13_pool', 0, 0, {}], ['batch_normalization_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 1536, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 423}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv1', 'inbound_nodes': [[['add_11', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv1_bn', 'inbound_nodes': [[['block14_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv1_act', 'inbound_nodes': [[['block14_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 2048, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 434}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv2', 'inbound_nodes': [[['block14_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv2_bn', 'inbound_nodes': [[['block14_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv2_act', 'inbound_nodes': [[['block14_sepconv2_bn', 0, 0, {}]]]}], 'input_layers': [['input_2', 0, 0]], 'output_layers': [['block14_sepconv2_act', 0, 0]]}, 'name': 'xception', 'inbound_nodes': [[['resizing', 0, 0, {}]]]}, {'class_name': 'Flatten', 'config': {'name': 'flatten', 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}, 'name': 'flatten', 'inbound_nodes': [[['xception', 1, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 64, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense', 'inbound_nodes': [[['flatten', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_4', 'trainable': True, 'dtype': 'float32', 'axis': [1], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_4', 'inbound_nodes': [[['dense', 0, 0, {}]]]}, {'class_name': 'ReLU', 'config': {'name': 're_lu', 'trainable': True, 'dtype': 'float32', 'max_value': None, 'negative_slope': array(0., dtype=float32), 'threshold': array(0., dtype=float32)}, 'name': 're_lu', 'inbound_nodes': [[['batch_normalization_4', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_1', 'inbound_nodes': [[['re_lu', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'classification_head_1', 'trainable': True, 'dtype': 'float32', 'activation': 'sigmoid'}, 'name': 'classification_head_1', 'inbound_nodes': [[['dense_1', 0, 0, {}]]]}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['classification_head_1', 0, 0]]}
    return murmur_config

def get_noise_model_configs(): 
    noise_input_config = {'batch_input_shape': (None, NOISE_IMAGE_SIZE[0], NOISE_IMAGE_SIZE[1], 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}
    noise_cast_to_float_config = {'name': 'cast_to_float32', 'trainable': True, 'dtype': 'float32'}
    noise_normalization_config = {'name': 'normalization', 'trainable': True, 'dtype': 'float32', 'axis': (-1,), 'mean': None, 'variance': None}
    noise_xception_config = {'name': 'xception', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, None, None, 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_2'}, 'name': 'input_2', 'inbound_nodes': []}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv1', 'trainable': True, 'dtype': 'float32', 'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv1', 'inbound_nodes': [[['input_2', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv1_bn', 'inbound_nodes': [[['block1_conv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv1_act', 'inbound_nodes': [[['block1_conv1_bn', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv2', 'trainable': True, 'dtype': 'float32', 'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv2', 'inbound_nodes': [[['block1_conv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv2_bn', 'inbound_nodes': [[['block1_conv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv2_act', 'inbound_nodes': [[['block1_conv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 21}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv1', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv1_bn', 'inbound_nodes': [[['block2_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block2_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block2_sepconv2_act', 'inbound_nodes': [[['block2_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 32}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv2', 'inbound_nodes': [[['block2_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv2_bn', 'inbound_nodes': [[['block2_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block2_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block2_pool', 'inbound_nodes': [[['block2_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization', 'inbound_nodes': [[['conv2d', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add', 'trainable': True, 'dtype': 'float32'}, 'name': 'add', 'inbound_nodes': [[['block2_pool', 0, 0, {}], ['batch_normalization', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv1_act', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 53}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv1', 'inbound_nodes': [[['block3_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv1_bn', 'inbound_nodes': [[['block3_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv2_act', 'inbound_nodes': [[['block3_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 64}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv2', 'inbound_nodes': [[['block3_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv2_bn', 'inbound_nodes': [[['block3_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_1', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block3_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block3_pool', 'inbound_nodes': [[['block3_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_1', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_1', 'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_1', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_1', 'inbound_nodes': [[['block3_pool', 0, 0, {}], ['batch_normalization_1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv1_act', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 85}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv1', 'inbound_nodes': [[['block4_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv1_bn', 'inbound_nodes': [[['block4_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv2_act', 'inbound_nodes': [[['block4_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 96}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv2', 'inbound_nodes': [[['block4_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv2_bn', 'inbound_nodes': [[['block4_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_2', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block4_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block4_pool', 'inbound_nodes': [[['block4_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_2', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_2', 'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_2', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_2', 'inbound_nodes': [[['block4_pool', 0, 0, {}], ['batch_normalization_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv1_act', 'inbound_nodes': [[['add_2', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 117}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv1', 'inbound_nodes': [[['block5_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv1_bn', 'inbound_nodes': [[['block5_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv2_act', 'inbound_nodes': [[['block5_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 128}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv2', 'inbound_nodes': [[['block5_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv2_bn', 'inbound_nodes': [[['block5_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv3_act', 'inbound_nodes': [[['block5_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 139}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv3', 'inbound_nodes': [[['block5_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv3_bn', 'inbound_nodes': [[['block5_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_3', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_3', 'inbound_nodes': [[['block5_sepconv3_bn', 0, 0, {}], ['add_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv1_act', 'inbound_nodes': [[['add_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 151}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv1', 'inbound_nodes': [[['block6_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv1_bn', 'inbound_nodes': [[['block6_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv2_act', 'inbound_nodes': [[['block6_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 162}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv2', 'inbound_nodes': [[['block6_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv2_bn', 'inbound_nodes': [[['block6_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv3_act', 'inbound_nodes': [[['block6_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 173}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv3', 'inbound_nodes': [[['block6_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv3_bn', 'inbound_nodes': [[['block6_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_4', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_4', 'inbound_nodes': [[['block6_sepconv3_bn', 0, 0, {}], ['add_3', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv1_act', 'inbound_nodes': [[['add_4', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 185}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv1', 'inbound_nodes': [[['block7_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv1_bn', 'inbound_nodes': [[['block7_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv2_act', 'inbound_nodes': [[['block7_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 196}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv2', 'inbound_nodes': [[['block7_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv2_bn', 'inbound_nodes': [[['block7_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv3_act', 'inbound_nodes': [[['block7_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 207}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv3', 'inbound_nodes': [[['block7_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv3_bn', 'inbound_nodes': [[['block7_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_5', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_5', 'inbound_nodes': [[['block7_sepconv3_bn', 0, 0, {}], ['add_4', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv1_act', 'inbound_nodes': [[['add_5', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 219}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv1', 'inbound_nodes': [[['block8_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv1_bn', 'inbound_nodes': [[['block8_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv2_act', 'inbound_nodes': [[['block8_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 230}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv2', 'inbound_nodes': [[['block8_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv2_bn', 'inbound_nodes': [[['block8_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv3_act', 'inbound_nodes': [[['block8_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 241}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv3', 'inbound_nodes': [[['block8_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv3_bn', 'inbound_nodes': [[['block8_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_6', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_6', 'inbound_nodes': [[['block8_sepconv3_bn', 0, 0, {}], ['add_5', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv1_act', 'inbound_nodes': [[['add_6', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 253}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv1', 'inbound_nodes': [[['block9_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv1_bn', 'inbound_nodes': [[['block9_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv2_act', 'inbound_nodes': [[['block9_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 264}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv2', 'inbound_nodes': [[['block9_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv2_bn', 'inbound_nodes': [[['block9_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv3_act', 'inbound_nodes': [[['block9_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 275}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv3', 'inbound_nodes': [[['block9_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv3_bn', 'inbound_nodes': [[['block9_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_7', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_7', 'inbound_nodes': [[['block9_sepconv3_bn', 0, 0, {}], ['add_6', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv1_act', 'inbound_nodes': [[['add_7', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 287}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv1', 'inbound_nodes': [[['block10_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv1_bn', 'inbound_nodes': [[['block10_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv2_act', 'inbound_nodes': [[['block10_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 298}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv2', 'inbound_nodes': [[['block10_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv2_bn', 'inbound_nodes': [[['block10_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv3_act', 'inbound_nodes': [[['block10_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 309}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv3', 'inbound_nodes': [[['block10_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv3_bn', 'inbound_nodes': [[['block10_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_8', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_8', 'inbound_nodes': [[['block10_sepconv3_bn', 0, 0, {}], ['add_7', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv1_act', 'inbound_nodes': [[['add_8', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 321}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv1', 'inbound_nodes': [[['block11_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv1_bn', 'inbound_nodes': [[['block11_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv2_act', 'inbound_nodes': [[['block11_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 332}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv2', 'inbound_nodes': [[['block11_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv2_bn', 'inbound_nodes': [[['block11_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv3_act', 'inbound_nodes': [[['block11_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 343}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv3', 'inbound_nodes': [[['block11_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv3_bn', 'inbound_nodes': [[['block11_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_9', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_9', 'inbound_nodes': [[['block11_sepconv3_bn', 0, 0, {}], ['add_8', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv1_act', 'inbound_nodes': [[['add_9', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 355}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv1', 'inbound_nodes': [[['block12_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv1_bn', 'inbound_nodes': [[['block12_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv2_act', 'inbound_nodes': [[['block12_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 366}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv2', 'inbound_nodes': [[['block12_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv2_bn', 'inbound_nodes': [[['block12_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv3_act', 'inbound_nodes': [[['block12_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 377}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv3', 'inbound_nodes': [[['block12_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv3_bn', 'inbound_nodes': [[['block12_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_10', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_10', 'inbound_nodes': [[['block12_sepconv3_bn', 0, 0, {}], ['add_9', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv1_act', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 389}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv1', 'inbound_nodes': [[['block13_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv1_bn', 'inbound_nodes': [[['block13_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv2_act', 'inbound_nodes': [[['block13_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 400}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv2', 'inbound_nodes': [[['block13_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv2_bn', 'inbound_nodes': [[['block13_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_3', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_3', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block13_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block13_pool', 'inbound_nodes': [[['block13_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_3', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_3', 'inbound_nodes': [[['conv2d_3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_11', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_11', 'inbound_nodes': [[['block13_pool', 0, 0, {}], ['batch_normalization_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 1536, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 420}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv1', 'inbound_nodes': [[['add_11', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv1_bn', 'inbound_nodes': [[['block14_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv1_act', 'inbound_nodes': [[['block14_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 2048, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 431}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv2', 'inbound_nodes': [[['block14_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv2_bn', 'inbound_nodes': [[['block14_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv2_act', 'inbound_nodes': [[['block14_sepconv2_bn', 0, 0, {}]]]}], 'input_layers': [['input_2', 0, 0]], 'output_layers': [['block14_sepconv2_act', 0, 0]]}
    noise_global_average_config = {'name': 'global_average_pooling2d', 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last', 'keepdims': False}
    noise_dense_config = {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
    noise_classification_head_config = {'name': 'classification_head_1', 'trainable': True, 'dtype': 'float32', 'activation': 'sigmoid'}
    
    return noise_input_config, noise_cast_to_float_config, noise_normalization_config, noise_xception_config, noise_global_average_config, noise_dense_config, noise_classification_head_config

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
