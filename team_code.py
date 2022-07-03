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
from multiprocessing import Pool
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
import uuid
import shutil
import tensorflow_datasets as tfds
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
from sklearn.utils import class_weight
import autokeras as ak
import random
from tensorflow import feature_column

tf.keras.utils.set_random_seed(42)
tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

################################################################################
#
# Audio preprocessing
#
################################################################################

import json
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from tqdm import tqdm
import json
from keras.engine.functional import Functional
import tensorflow_addons as tfa
import keras_tuner as kt

from autokeras.engine import node as node_module
from autokeras.engine import tuner
from autokeras.nodes import Input
from tensorflow import nest
from tensorflow import keras
from pathlib import Path
from typing import List
from typing import Optional
from typing import Type
from autokeras import blocks
from typing import Union
from autokeras.engine import head as head_module
from autokeras import graph
from autokeras import keras_layers
import tensorflow_decision_forests as tfdf


hop_length = 256
REAL_SR = 4000
FIGURE_SIZE = 3
EMBDS_PER_PATIENTS = 138
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


LOAD_TRAINED_MODELS = False
GENERATE_MEL_SPECTOGRAMS_TRAIN = True
TRAIN_NOISE_DETECTION = True

NOISE_IMAGE_SIZE = [108, 108]
RESHUFFLE_PATIENT_EMBS_N = 5
MURMUR_IMAGE_SIZE = deepcopy(NOISE_IMAGE_SIZE)
# class_weight_murmur = {0: 1, 1: 1}
# class_weight_decision = {0: 1, 1: 1.5}  
MAX_QUEUE = 50000
batch_size_murmur = 256
RUN_AUTOKERAS_NOISE = False
RUN_AUTOKERAS_MURMUR = False
RUN_AUTOKERAS_DECISION = False
FINAL_TRAINING = True
USE_COMPLEX_MODELS = True
# EMBEDDING_LAYER_REFERENCE_MURMUR_MODEL = -1 if not USE_COMPLEX_MODELS else -2
EMBEDDING_LAYER_REFERENCE_MURMUR_MODEL = -1

WORKERS = min(os.cpu_count() - 1, 8)

N_DECISION_LAYERS = 2

OHH_ARGS = None
RUN_TEST = None

TRAIN_FRAC_lbl = "TRAIN_FRAC"
EMBDS_PER_PATIENTS_lbl = "EMBDS_PER_PATIENTS"
VAL_FRAC_MURMUR_lbl = "VAL_FRAC_MURMUR"
NOISE_IMAGE_SIZE_lbl = "NOISE_IMAGE_SIZE"
RESHUFFLE_PATIENT_EMBS_N_lbl = "RESHUFFLE_PATIENT_EMBS_N"
MURMUR_IMAGE_SIZE_lbl = "MURMUR_IMAGE_SIZE"
class_weight_murmur_lbl = "class_weight_murmur"
class_weight_decision_lbl = "class_weight_decision"
batch_size_murmur_lbl = "batch_size_murmur"
EMBS_SIZE_lbl = "EMBS_SIZE"
CNN_MURMUR_MODEL_lbl = "CNN_MURMUR_MODEL"
N_DECISION_LAYERS_lbl = "N_DECISION_LAYERS"
NEURONS_DECISION_lbl = "NEURONS_DECISION"
IS_DROPOUT_IN_DECISION_lbl = "DROPOUT_IN_DECISION"
DROPOUT_VALUE_IN_DECISION_lbl = "DROPOUT_VALUE_IN_DECISION"
IS_MURMUR_MODEL_XCEPTION_lbl = "IS_MURMUR_MODEL_XCEPTION"
N_MURMUR_CNN_NEURONS_LAYERS_lbl = "N_MURMUR_CNN_NEURONS_LAYERS"
DROPOUT_VALUE_IN_MURMUR_lbl = "DROPOUT_VALUE_IN_MURMUR"
IS_DROPOUT_IN_MURMUR_lbl = "DROPOUT_IN_MURMUR"
N_MURMUR_LAYERS_lbl = "N_MURMUR_LAYERS"
STEPS_PER_EPOCH_DECISION_lbl = "STEPS_PER_EPOCH_DECISION"
UNKOWN_RANDOM_MIN_THRESHOLD_lbl = "UNKOWN_RANDOM_MIN_THRESHOLD"
BATCH_SIZE_DECISION_lbl = "BATCH_SIZE_DECISION"
IMG_HEIGHT_RATIO_lbl = "IMG_HEIGHT_RATIO_lbl"
ALGORITHM_HPS = {
    TRAIN_FRAC_lbl : 0.3,
    IMG_HEIGHT_RATIO_lbl : 1,
    STEPS_PER_EPOCH_DECISION_lbl : None,
    EMBDS_PER_PATIENTS_lbl : 50,
    VAL_FRAC_MURMUR_lbl : 0.3,
    NOISE_IMAGE_SIZE_lbl : 108,
    RESHUFFLE_PATIENT_EMBS_N_lbl : 4,
    MURMUR_IMAGE_SIZE_lbl : 108,
    class_weight_murmur_lbl : 5,
    class_weight_decision_lbl : 5,
    batch_size_murmur_lbl : 32,
    EMBS_SIZE_lbl : 2,
    CNN_MURMUR_MODEL_lbl : True,
    N_DECISION_LAYERS_lbl : 1,
    NEURONS_DECISION_lbl : 8,
    IS_DROPOUT_IN_DECISION_lbl : False,
    DROPOUT_VALUE_IN_DECISION_lbl : 0.2,
    IS_MURMUR_MODEL_XCEPTION_lbl : False,
    N_MURMUR_CNN_NEURONS_LAYERS_lbl : 64,
    DROPOUT_VALUE_IN_MURMUR_lbl : 0.25,
    N_MURMUR_LAYERS_lbl : 2,
    IS_DROPOUT_IN_MURMUR_lbl : True,
    UNKOWN_RANDOM_MIN_THRESHOLD_lbl : 0.8,
    BATCH_SIZE_DECISION_lbl : 64
}

from tensorboard.plugins.hparams import api as hp

if os.path.exists("ohh.config"):
    import boto3
    OHH_ARGS = json.loads(open("ohh.config", "r").read().strip())
    if ("AWS_ID" in OHH_ARGS) and ("AWS_PASS" in OHH_ARGS):
        s3 = boto3.client("s3",  aws_access_key_id=OHH_ARGS["AWS_ID"], aws_secret_access_key=OHH_ARGS["AWS_PASS"])
    RUN_TEST  = OHH_ARGS["TEST_MODE"]
    if "embs_size" in OHH_ARGS:
        ALGORITHM_HPS[EMBS_SIZE_lbl] =  OHH_ARGS["embs_size"]
    if "reshuffle_patient" in OHH_ARGS:
        ALGORITHM_HPS[RESHUFFLE_PATIENT_EMBS_N_lbl] =  OHH_ARGS["reshuffle_patient"]
    if "embs_per_patient" in OHH_ARGS:
        ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl] =  OHH_ARGS["embs_per_patient"]
    if "murmur_image_size" in OHH_ARGS:
        ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl] =  OHH_ARGS["murmur_image_size"]
    if "weight_class_murmur" in OHH_ARGS:
        ALGORITHM_HPS[class_weight_murmur_lbl] =  OHH_ARGS["weight_class_murmur"]
    if "weight_class_decisions" in OHH_ARGS:
        ALGORITHM_HPS[class_weight_decision_lbl] =  OHH_ARGS["weight_class_decisions"]
    
logger.info("Embs Size: {}" .format(ALGORITHM_HPS[EMBS_SIZE_lbl]))
logger.info("Weight class murmur: {}" .format(ALGORITHM_HPS[class_weight_murmur_lbl]))
logger.info("Weight class decision: {}" .format(ALGORITHM_HPS[class_weight_decision_lbl]))
logger.info("Murmur Image Size: {}" .format(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]))
logger.info("Random embeddings per patient: {}" .format(ALGORITHM_HPS[RESHUFFLE_PATIENT_EMBS_N_lbl]))
logger.info("Reshuffle for training: {}" .format(ALGORITHM_HPS[RESHUFFLE_PATIENT_EMBS_N_lbl]))


    # Embs Size : [16, 64, 256]
    # Weight class murmur : [1, 1.5, 3, 5]
    # Weight class decision : [1, 1.5, 3, 5]
    # Murmur Image Size : [32, 72, 108, 216]
    # Random embeddings per patient: [15, 50, 100, 128]
    # Reshuffle for training: [1, 3, 5, 10]

if RUN_TEST:
    logger.info("Running test")
    MURMUR_EPOCHS = 1
    NOISE_EPOCHS = 1
    MURMUR_DECISION_EPOCHS = 1
    MAX_TRIALS = 1
else:
    logger.info("Running full")
    MURMUR_EPOCHS = 1000
    NOISE_EPOCHS = 100
    MURMUR_DECISION_EPOCHS = 1000
    MAX_TRIALS = 100

#Download autoencoder
enc = OneHotEncoder()
enc.fit([[True], [False]])

# Main hyper_parameters
# Embs Size : [16, 64, 256]
# Weight class murmur : [1, 1.5, 3, 5]
# Weight class decision : [1, 1.5, 3, 5]
# Murmur Image Size : [32, 72, 108, 216]
# Random embeddings per patient: [15, 50, 100, 128]
# Reshuffle for training: [1, 3, 5, 10]

class OHHGraph(graph.Graph):
  def _compile_keras_model(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer_name = hp.Choice(
            "optimizer",
            ["adam"],
            default="adam",
        )
        # TODO: add adadelta optimizer when it can optimize embedding layer on GPU.
        learning_rate = hp.Choice(
            "learning_rate", [1e-3, 1e-4], default=1e-3
        )

        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == "adam_weight_decay":
            steps_per_epoch = int(self.num_samples / self.batch_size)
            num_train_steps = steps_per_epoch * self.epochs
            warmup_steps = int(
                self.epochs * self.num_samples * 0.1 / self.batch_size
            )

            lr_schedule = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=num_train_steps,
                end_learning_rate=0.0,
            )
            if warmup_steps:
                lr_schedule = keras_layers.WarmUp(
                    initial_learning_rate=learning_rate,
                    decay_schedule_fn=lr_schedule,
                    warmup_steps=warmup_steps,
                )

            optimizer = keras_layers.AdamWeightDecay(
                learning_rate=lr_schedule,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )

        model.compile(
            optimizer=optimizer, metrics=self._get_metrics(), loss=self._get_loss()
        )

        return model

class OHHAutoModel(ak.AutoModel):

  def _assemble(self):
        """Assemble the Blocks based on the input output nodes."""
        inputs = nest.flatten(self.inputs)
        outputs = nest.flatten(self.outputs)

        middle_nodes = [input_node.get_block()(input_node) for input_node in inputs]

        # Merge the middle nodes.
        if len(middle_nodes) > 1:
            output_node = blocks.Merge()(middle_nodes)
        else:
            output_node = middle_nodes[0]

        outputs = nest.flatten(
            [output_blocks(output_node) for output_blocks in outputs]
        )
        return OHHGraph(inputs=inputs, outputs=outputs)

  def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            graph = OHHGraph(inputs=self.inputs, outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, head_module.Head) for output in self.outputs]):
            # Clear session to reset get_uid(). The names of the blocks will
            # start to count from 1 for new blocks in a new AutoModel afterwards.
            # When initializing multiple AutoModel with Task API, if not
            # counting from 1 for each of the AutoModel, the predefined hp
            # values in task specifiec tuners would not match the names.
            keras.backend.clear_session()
            graph = self._assemble()
            self.outputs = graph.outputs
            keras.backend.clear_session()

        return graph
    
# Load a WAV file.
def load_wav_file_ohh(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    recording = recording / 2 ** (16 - 1)
    return recording, frequency

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def clean_frequency_heartbeat(wavfile_path):
  x, sr_fake = load_wav_file_ohh(wavfile_path)
  sr = REAL_SR
  min_f = 20
#   max_f = 786 #Reference: https://bmcpediatr.biomedcentral.com/track/pdf/10.1186/1471-2431-7-23.pdf
  max_f = 530 #Reference: https://bmcpediatr.biomedcentral.com/track/pdf/10.1186/1471-2431-7-23.pdf
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

def generate_mel_image_v2(x, sr, seconds_window, hop_length=128):
  total_duration = len(x)/sr 
  fig, ax = plt.subplots(nrows=1, sharex=True, figsize=(FIGURE_SIZE * total_duration / seconds_window, FIGURE_SIZE))
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
  return fig, ax

def generate_mel_wav_crops(filepath_output_folder):
    filepath, output_folder = filepath_output_folder
    seconds_window = 2
    # print(filepath)
    if output_folder:
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

def generate_mel_wav_crops_v2(filepath_output_folder):
    filepath, output_folder = filepath_output_folder
    seconds_window = 2
    # print(filepath)
    if output_folder:
        x, sr_fake = librosa.load(filepath, sr=REAL_SR)
        sr = REAL_SR
        duration_seconds = len(x)/sr
        fig_image, ax_image = generate_mel_image_v2(x, sr, seconds_window, hop_length=32)
        ax_width = ax_image.get_xlim()[1]
        fig_image.canvas.draw()
        temp_canvas = fig_image.canvas
        pil_image_full = PIL.Image.frombytes('RGB', temp_canvas.get_width_height(),  temp_canvas.tostring_rgb())
        pil_image_width, pil_image_height = pil_image_full.size
        for end_time in range(seconds_window, math.floor(duration_seconds), seconds_window):
            # Crop sound
            start_time = end_time - seconds_window
            image_begin = (pil_image_width / duration_seconds) * start_time
            image_end = image_begin + (pil_image_width / duration_seconds) * (seconds_window) 
            # x_cut = x[start_time*sr: start_time*sr + seconds_window*sr]
            pil_image = pil_image_full.crop((image_begin, pil_image_height * (1-ALGORITHM_HPS[IMG_HEIGHT_RATIO_lbl]), image_end, pil_image_height))
            # pil_image = generate_mel_image(x_cut, sr, hop_length=32)
            output_filepath_prefix = output_folder.rstrip("/") + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + "_{}_to_{}".format(int(start_time), int(end_time)) 
            output_filepath_wav =  output_filepath_prefix + ".wav" 
            output_filepath_image = output_filepath_prefix + ".png" 
            # librosa.output.write_wav(filepath, x_cut, sr, norm=True)
            # x_cut = librosa.util.normalize(x_cut)
            pil_image.save(output_filepath_image)
            
            # sf.write(output_filepath_wav, x_cut, sr, 'PCM_16')
            pil_image.close()
            del pil_image
        fig_image.clf()
        plt.close()
        del x
        del sr

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
        ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl] = 72
        ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl] = 72
        destiny_zip = os.path.join(model_dir, 'murmur_model.zip')
        destiny_model = os.path.join(model_dir, 'murmur_model.model')
        url = "http://algodev.matheusaraujo.com:8888/pretrained_models/model_murmur_checkpoint72_x_72.zip"
        
    if model_type == "noise":
        ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl] = 108
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
                
NOISE_DETECTION_WORKING_DIR = "./" 
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

def delete_murmur_and_embs_folders():
    shutil.rmtree(train_folder_murmur, ignore_errors=True)
    shutil.rmtree(test_folder_murmur, ignore_errors=True)
    shutil.rmtree(val_folder_murmur, ignore_errors=True)
    shutil.rmtree(train_outcome_folder, ignore_errors=True)
    shutil.rmtree(test_outcome_folder, ignore_errors=True)
    shutil.rmtree(val_outcome_folder, ignore_errors=True)
    shutil.rmtree(train_embs_folder_murmur, ignore_errors=True)
    shutil.rmtree(test_embs_folder_murmur, ignore_errors=True)
    shutil.rmtree(val_embs_folder_murmur, ignore_errors=True)
    

def clean_current_path():
    shutil.rmtree("recordings_aux", ignore_errors=True)
    shutil.rmtree("image_aux", ignore_errors=True)
    delete_murmur_and_embs_folders()

def prepare_train_val_test_murmur_folders():    
    delete_murmur_and_embs_folders()

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
  image = tf.keras.preprocessing.image.load_img(filepath, target_size=ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl])
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
    if embs_df.shape[0] < ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl]:
        embs_df = embs_df.sample(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl], replace=True, random_state=42)
    else:
        embs_df = embs_df.head(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl])
    destiny_file = destiny_folder + os.path.sep + patient_id
    embs_df.to_pickle(destiny_file)
    # logger.info("Saved: " + destiny_file)

def generate_patient_embeddings_folder_v2(patient_id, split, label, patient_embs):
    # destiny_folder = split + "_embs_folder_murmur" + os.path.sep +  label
    # original_files_query = split + "_folder_murmur" + os.path.sep + label + os.path.sep + patient_id + "*"
    # patient_files = glob.glob(original_files_query)
    embs_df = patient_embs.sample(frac=1, random_state=42)
    # We lose potential embds here, fix later
    if embs_df.shape[0] < ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl]:
        embs_df = embs_df.sample(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl], replace=True, random_state=42)
    else:
        embs_df = embs_df.head(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl])
    # destiny_file = destiny_folder + os.path.sep + patient_id
    # embs_df.to_pickle(destiny_file)
    # logger.info("Saved: " + destiny_file)
    return embs_df
    

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

class ComputedWeightedAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name='computed_weighted_accuracy', **kwargs):
    super(ComputedWeightedAccuracy, self).__init__(name=name, **kwargs)
    self.computed_weighted_accuracy = self.add_weight(name='cwa', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    cwa = compute_weighted_accuracy(y_true, y_pred, classes = ["Abnormal", "Normal"])
    self.computed_weighted_accuracy.assign(cwa)
  def result(self):
    return self.computed_weighted_accuracy

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
    try:
        label = patient_murmur_outcome_df[patient_murmur_outcome_df["patient_id"] == patient_id].iloc[0][problem]
    except:            
        import ipdb;ipdb.set_trace()
        pass
    label = True if label == "Present" or label == "Abnormal" else False
    embs.append(np.vstack(patient_df.values).flatten())
    labels.append(label)
  return embs, labels


# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    AUX_FOLDER = "recordings_aux"
    AUX_IMGS_FOLDER = "images_aux"
    AUX_IMGS_POSITIVE_FOLDER = os.path.join(AUX_IMGS_FOLDER, "positive")
    AUX_IMGS_NEGATIVE_FOLDER = os.path.join(AUX_IMGS_FOLDER, "negative")
    murmur_image_folders = [train_positive_folder, train_negative_folder, val_positive_folder, val_negative_folder, test_positive_folder, test_negative_folder]
    global GENERATE_MEL_SPECTOGRAMS_TRAIN, TRAIN_NOISE_DETECTION
    if GENERATE_MEL_SPECTOGRAMS_TRAIN:
        clean_current_path()
    
    os.makedirs(AUX_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_POSITIVE_FOLDER, exist_ok=True)
    os.makedirs(AUX_IMGS_NEGATIVE_FOLDER, exist_ok=True)
    
    
    
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    os.system("tar -xf noise_detection_sandbox.tar.gz -C {}".format(NOISE_DETECTION_WORKING_DIR))

    if LOAD_TRAINED_MODELS:
        try:
            noise_model = load_pretrained_model(model_folder, "noise")
            logger.info("Loading model: {}".format("noises"))
            # noise_model = load_model(os.path.join(model_folder, "noise_model.tf"), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
            logger.info(noise_model.summary())
            # murmur_model = load_model(os.path.join(model_folder, "murmur_model.tf"), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
            murmur_model = load_pretrained_model(model_folder, "murmur")
            logger.info(murmur_model.summary())
            # murmur_decision_model = load_model(os.path.join(model_folder, "murmur_decision.tf"), custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
            murmur_decision_model = load_pretrained_model(model_folder, "murmur_decision")
            logger.info(murmur_decision_model.summary())
            # save_challenge_model(model_folder, noise_model, murmur_model, murmur_decision_model)
            if verbose >= 1:
                print('Training completed.')
        except OSError:
            logger.error("Could not load models setting all training to True")
            TRAIN_NOISE_DETECTION = True
            GENERATE_MEL_SPECTOGRAMS_TRAIN = True

    # Noise model - Parameters found after runnign AutoKeras
    if TRAIN_NOISE_DETECTION:
        batch_size = 32
         
        
        if RUN_AUTOKERAS_NOISE:
            # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            # filepath=os.path.join(model_folder, "noise_model_ak.model"),
            # save_weights_only=False,
            # monitor='val_auc',
            # loss="binary_crossentropy",
            # mode='max',
            # initial_value_threshold=0.8,
            # save_best_only=True)
            noise_detection_dataset_train = ak.image_dataset_from_directory(
                NOISE_DETECTION_IMGS_PATH,
                # Use 20% data as testing data.
                validation_split=1 - ALGORITHM_HPS[TRAIN_FRAC_lbl],
                subset="training",
                # Set seed to ensure the same split when loading testing data.
                seed=42,
                image_size=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl]),
                batch_size=batch_size,
            )

            noise_detection_dataset_test = ak.image_dataset_from_directory(
                NOISE_DETECTION_IMGS_PATH,
                validation_split=1 - ALGORITHM_HPS[TRAIN_FRAC_lbl],
                subset="validation",
                seed=42,
                image_size=ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl],
                batch_size=batch_size,
            )
            
            input_node = ak.ImageInput()
            output_node = ak.ImageBlock(
                block_type="xception",
                augment=False
            )(input_node)
            output_node = ak.ClassificationHead()(output_node)

            clf = OHHAutoModel(
                inputs=input_node, seed=42, objective=kt.Objective("val_auc", direction="max"), outputs=output_node, overwrite=True, 
                max_trials=MAX_TRIALS, metrics = get_all_metrics()
            )
            clf.fit(noise_detection_dataset_train, epochs = NOISE_EPOCHS, workers= WORKERS, max_queue_size=MAX_QUEUE, use_multiprocessing=False)
            
            #TODO: Test
            logger.info("Noise Model Classification Report")
            logger.info(clf.evaluate(noise_detection_dataset_test, return_dict=True))
            
            noise_model_new = keras.models.load_model(clf.tuner.best_model_path, custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
            
 
        else:
            noise_detection_dataset_train_val = tf.keras.utils.image_dataset_from_directory(NOISE_DETECTION_IMGS_PATH, subset="training", validation_split=1 - ALGORITHM_HPS[TRAIN_FRAC_lbl], seed=42, image_size=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl],ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl]) )
            dataset_size = len(noise_detection_dataset_train_val)
            noise_detection_dataset_train = noise_detection_dataset_train_val.take(int(dataset_size * ALGORITHM_HPS[TRAIN_FRAC_lbl]))
            noise_detection_dataset_val = noise_detection_dataset_train_val.skip(int(dataset_size * ALGORITHM_HPS[TRAIN_FRAC_lbl]))
            noise_detection_dataset_test = tf.keras.utils.image_dataset_from_directory(NOISE_DETECTION_IMGS_PATH, subset="validation", validation_split=1 - ALGORITHM_HPS[TRAIN_FRAC_lbl], seed=42, image_size=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl]))
            
               
            # noise_detection_dataset_train = noise_detection_dataset_train.batch(batch_size,drop_remainder=True)        
            # noise_detection_dataset_val = noise_detection_dataset_val.batch(batch_size,drop_remainder=True)        
            # noise_detection_dataset_test = noise_detection_dataset_test.batch(batch_size,drop_remainder=True)   
            
            noise_detection_dataset_train = noise_detection_dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            noise_detection_dataset_val = noise_detection_dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            noise_detection_dataset_test = noise_detection_dataset_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            if USE_COMPLEX_MODELS:
                # noise_model_new = get_noise_model()
                noise_model_new = get_noise_model_v2()
                
            else:
                noise_model_new = tf.keras.models.Sequential()
                noise_model_new.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], 3)))
                noise_model_new.add(tf.keras.layers.MaxPooling2D((2, 2)))
                noise_model_new.add(tf.keras.layers.Dropout(.2))
                noise_model_new.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], 3)))
                noise_model_new.add(tf.keras.layers.MaxPooling2D((2, 2)))
                noise_model_new.add(tf.keras.layers.Flatten())
                noise_model_new.add(tf.keras.layers.Dropout(.5))
                noise_model_new.add(tf.keras.layers.Dense(ALGORITHM_HPS[EMBS_SIZE_lbl], activation='relu'))
                noise_model_new.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                noise_model_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.0001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), 
                        loss="binary_crossentropy",
                        metrics=get_all_metrics())
            early_stopping_noise = tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                min_delta=0.0001,
                patience=10,
                verbose=1,
                mode="max",
                baseline=None,
                restore_best_weights=True,
            )
            noise_model_new.fit(noise_detection_dataset_train, batch_size = batch_size, max_queue_size=MAX_QUEUE, epochs = NOISE_EPOCHS, callbacks=[early_stopping_noise], validation_data=noise_detection_dataset_val, workers= WORKERS)

            logger.info("Noise Model Classification Report")
            logger.info(noise_model_new.evaluate(noise_detection_dataset_test, return_dict=True))
    else:
        noise_model_new = noise_model
        
    tf.keras.models.save_model(
            noise_model_new,
            os.path.join(model_folder, 'noise_model.tf'),
            overwrite=True,
            include_optimizer=False,
            save_format="tf",
            signatures=None,
            options=None,
            save_traces=True
        )
    
    
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
    test_set = unique_original_files.sample(frac=1-ALGORITHM_HPS[TRAIN_FRAC_lbl], random_state=42)
    train_val_set = unique_original_files[~unique_original_files.isin(test_set)]
    train_set = train_val_set.sample(frac=(1 - ALGORITHM_HPS[VAL_FRAC_MURMUR_lbl]), random_state=42)
    val_set = train_val_set[~train_val_set.isin(train_set)]
    
    if GENERATE_MEL_SPECTOGRAMS_TRAIN:
        for i in tqdm(range(num_patient_files)):
        #     # Load the current patient data and recordings.
            current_patient_data = load_patient_data(patient_files[i])
            current_recordings, locations = load_recordings_custom(data_folder, current_patient_data)
            murmur_locations = get_murmur_locations(current_patient_data)
            patient_id = get_patient_id(current_patient_data)
            split = ""
            if (train_set == patient_id).any():
                split = "train"
            elif (val_set == patient_id).any():
                split = "val"
            elif (test_set == patient_id).any():
                split = "test"
            else:
                raise Exception("Should never happen.")
            
                
            for recording, location in zip(current_recordings, locations):
                recording = recording / 2 ** (16 - 1) # Normalize as: https://stackoverflow.com/questions/50062358/difference-between-load-of-librosa-and-read-of-scipy-io-wavfile
                recording_name = str(patient_id) + "_" + get_unique_name()
                patients_file_informations.append({
                    "file_prefix" : recording_name,
                    "murmur" : get_murmur(current_patient_data),
                    "outcome" : get_outcome(current_patient_data),
                    "murmur_locations" : murmur_locations,
                    "pregnancy_status" : get_pregnancy_status(current_patient_data),
                    "weight" : get_weight(current_patient_data),
                    "height" : get_height(current_patient_data),
                    "sex" : get_sex(current_patient_data),
                    "age" :  get_age(current_patient_data),
                    "split" : split,
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
        destiny_folders = []
        for clean_file in tqdm(clean_files): 
            # TODO: This should be paralelized
            file_prefix = os.path.basename(clean_file).split("_clean")[0]
            file_info = patients_file_informations_df[patients_file_informations_df['file_prefix'] == file_prefix].iloc[0]
            destiny_folder = None
            if file_info["murmur"] == "Present" and file_info["is_murmur_location"]:
                if file_info["patient_id"] in train_set.values:
                    destiny_folder = train_positive_folder
                if file_info["patient_id"] in val_set.values:
                    destiny_folder = val_positive_folder
                if file_info["patient_id"] in test_set.values:
                    destiny_folder = test_positive_folder
            if file_info["murmur"] == "Unknown" and random.random() > ALGORITHM_HPS[UNKOWN_RANDOM_MIN_THRESHOLD_lbl]:
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
            destiny_folders.append(destiny_folder)      
        filepath_output_folder = zip(clean_files, destiny_folders)
        # for args_filepath in tqdm(filepath_output_folder):
        #     generate_mel_wav_crops_v2(args_filepath)
        pool = Pool(processes=(min(WORKERS, 8)))
        for _ in tqdm(pool.imap(generate_mel_wav_crops_v2, filepath_output_folder), total=len(destiny_folders)):
            pass
        pool.close()

    # Clean imgs with noise prediction
    
    noise_murmur_df_list = []
    for folder in murmur_image_folders:
        try:
            noise_detection_dataset = tf.keras.utils.image_dataset_from_directory(folder, label_mode = None, image_size=ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], shuffle=False)
        except ValueError:
            continue
        predictions = noise_model_new.predict(noise_detection_dataset)
        noise_murmur_df_list.append(pd.DataFrame({"predictions" : predictions.flatten(), "filepath" : noise_detection_dataset.file_paths}))
    
    if len(noise_murmur_df_list) > 0:
        logger.info("Files to remove noise: {}".format(len(noise_murmur_df_list)))
        logger.info("First 10: {}".format(pd.Series(noise_murmur_df_list).head(10)))
        noise_murmur_df = pd.concat(noise_murmur_df_list)        
        files_to_exclude = noise_murmur_df[noise_murmur_df["predictions"] > 0.5]
        logger.info("Remove noisy files")
        if not RUN_TEST and files_to_exclude.shape[0] > 0:
            for filepath in tqdm(files_to_exclude["filepath"]):
                os.remove(filepath)
    else:
        logger.info("No files to remove noise.")
    
    # Dataset for final decision
    all_files = []
    for folder in murmur_image_folders:
        all_files = all_files + glob.glob("{}/*".format(folder))
    
    patient_ids = pd.Series(all_files).str.split("/").str[2].str.split("_").str[0]
    split = pd.Series(all_files).str.split("/").str[0].str.split("_").str[0]
    label = pd.Series(all_files).str.split("/").str[1]
    patient_split_df = pd.concat([patient_ids, split, label],axis=1)
    patient_split_df.columns = ["patient_id", "split", "label"]
    patient_split_df = patient_split_df.drop_duplicates(subset=["patient_id"])
    
    # Load dataset for murmur model training  
    
    murmur_model_dataset_train = tf.keras.utils.image_dataset_from_directory(train_folder_murmur, label_mode="binary", batch_size=ALGORITHM_HPS[batch_size_murmur_lbl], seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]) )
    murmur_murmur_dataset_val = tf.keras.utils.image_dataset_from_directory(val_folder_murmur, label_mode="binary", batch_size=ALGORITHM_HPS[batch_size_murmur_lbl], seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]) )
    murmur_murmur_dataset_test = tf.keras.utils.image_dataset_from_directory(test_folder_murmur, label_mode="binary", batch_size=ALGORITHM_HPS[batch_size_murmur_lbl], seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]), )
       
    sklearn_weights_murmur = class_weight.compute_class_weight("balanced",classes=[False, True], y= (np.vstack(murmur_model_dataset_train.map(lambda x,y: y)) == 1).reshape(1,-1)[0].tolist())
    sklearn_weights_murmur = dict(enumerate(sklearn_weights_murmur))
    sklearn_weights_murmur[1] *= ALGORITHM_HPS[class_weight_murmur_lbl]
    
    if FINAL_TRAINING:
        murmur_model_dataset_train = murmur_model_dataset_train.concatenate(murmur_murmur_dataset_val)
        murmur_murmur_dataset_val = murmur_murmur_dataset_test
    
    if RUN_AUTOKERAS_MURMUR:
        input_node = ak.ImageInput()
        output_node = ak.XceptionBlock(pretrained=False)(input_node)
        output_node = ak.SpatialReduction(reduction_type="flatten")(output_node)
        output_node = ak.DenseBlock(num_layers=1, num_units=64, dropout=0)(output_node)
        output_node = ak.ClassificationHead(num_classes = 2, dropout=0)(output_node)

        clf = OHHAutoModel(
            inputs=input_node, tuner="bayesian", seed=42, objective=kt.Objective("val_auc", direction="max"), outputs=output_node, overwrite=True, 
            max_trials=MAX_TRIALS, metrics = get_all_metrics())
        clf.fit(murmur_model_dataset_train, validation_data=murmur_murmur_dataset_val, epochs = MURMUR_EPOCHS, class_weight = sklearn_weights_murmur, callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0,
            patience=20,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True
        )], workers= WORKERS)
        murmur_model_new = keras.models.load_model(clf.tuner.best_model_path, custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
  
    else:
        if USE_COMPLEX_MODELS:
            murmur_model_new = get_murmur_model()
            
        else:
            murmur_model_new = tf.keras.models.clone_model(noise_model_new)
            murmur_model_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 0.0001,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=get_all_metrics())
        
        murmur_model_new.fit(murmur_model_dataset_train, validation_data=murmur_murmur_dataset_val, 
                             epochs = MURMUR_EPOCHS, max_queue_size=MAX_QUEUE, class_weight=sklearn_weights_murmur, callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            min_delta=0,
            patience=20,
            verbose=0,
            mode="max",
            baseline=None,
            restore_best_weights=True
        )], workers= WORKERS)
    # murmur_model_new.set_weights(noise_model_new.get_weights())
    logger.info("Murmur Model Performance")
    logger.info(murmur_model_new.evaluate(murmur_murmur_dataset_test, return_dict=True))
    tf.keras.models.save_model(
            murmur_model_new,
            os.path.join(model_folder, 'murmur_model.tf'),
            overwrite=True,
            include_optimizer=True,
            save_format="tf",
            signatures=None,
            options=None,
            save_traces=True
        )
    
    murmur_embedding_model = tf.keras.models.Sequential(murmur_model_new.layers[:EMBEDDING_LAYER_REFERENCE_MURMUR_MODEL])
    
    # Generating embeddings
    logger.info("Loading all images...")
    murmur_model_dataset_train = tf.keras.utils.image_dataset_from_directory(train_folder_murmur, label_mode="binary", seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]), shuffle=False)
    train_filepaths = murmur_model_dataset_train.file_paths
    murmur_model_dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    murmur_model_dataset_val = tf.keras.utils.image_dataset_from_directory(val_folder_murmur, label_mode="binary", batch_size=1, seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl],ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]) , shuffle=False )
    val_filepaths = murmur_model_dataset_val.file_paths
    murmur_model_dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    murmur_model_dataset_test = tf.keras.utils.image_dataset_from_directory(test_folder_murmur, label_mode="binary", batch_size=1, seed=42, image_size=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]), shuffle=False )
    test_filepaths = murmur_model_dataset_test.file_paths
    murmur_model_dataset_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # murmur_model_dataset_all = murmur_model_dataset_train.concatenate(murmur_murmur_dataset_val).concatenate(murmur_murmur_dataset_test)
    # all_murmur_files_series = pd.Series(glob.glob(os.path.join(train_folder_murmur, "**/*.png")) + glob.glob(os.path.join(val_folder_murmur, "**/*.png")) + glob.glob(os.path.join(test_folder_murmur, "**/*.png")))
    # all_murmur_files_images = all_murmur_files_series.apply(load_image_array)
    logger.info("Predicting for all images...")
    all_file_paths = pd.Series(train_filepaths + val_filepaths + test_filepaths)
    all_murmur_files_embs = np.vstack((murmur_embedding_model.predict(murmur_model_dataset_train), murmur_embedding_model.predict(murmur_model_dataset_val), murmur_embedding_model.predict(murmur_model_dataset_test)))
    # del all_murmur_files_images
    patient_ids = all_file_paths.apply(lambda x: x.split("/")[2].split("_")[0])
    embs_df = pd.DataFrame({"filepath"  : all_file_paths, "embs" : all_murmur_files_embs.tolist(), "patient_id": patient_ids})
    
    embs_train = []
    embs_label_train = []
    embs_patient_id_train = []
    embs_val = []
    embs_label_val = []
    embs_patient_id_val = []
    embs_test = []
    embs_label_test = []
    embs_patient_id_test = []
    logger.info("Separeting sets for murmur decision...")
    for patient_id, patient_embs in tqdm(embs_df.groupby("patient_id")):
        patient_row = patient_split_df[patient_split_df["patient_id"] == patient_id].iloc[0]
        embs_df = patient_embs["embs"].sample(frac=1, random_state=42)
        label = patient_row["label"] == "positive"
        # We lose potential embds here, fix later
        if patient_row["split"] == "train":
            for repetition in range(ALGORITHM_HPS[RESHUFFLE_PATIENT_EMBS_N_lbl]):
                embs_df = embs_df.sample(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl], replace=True, random_state=42 + repetition)
                embs_train.append(np.vstack(embs_df).flatten())
                embs_label_train.append(label)
                embs_patient_id_train.append(patient_id)
        # We losing info in the validation
        if embs_df.shape[0] < ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl]:
            embs_df = embs_df.sample(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl], replace=True, random_state=42)
        else:
            embs_df = embs_df.head(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl])
        if patient_row["split"] == "val":
            embs_val.append(np.vstack(embs_df).flatten())
            embs_label_val.append(label)
            embs_patient_id_val.append(patient_id)
        if patient_row["split"] == "test":
            embs_test.append(np.vstack(embs_df).flatten())
            embs_label_test.append(label)
            embs_patient_id_test.append(patient_id)
        
    # generate_patient_embeddings_folder_v2(patient_id, patient_row["split"], patient_row["label"], patient_embs["embs"])
    # embs_train, labels_train = load_embs_labels(train_embs_folder_murmur, "murmur", patient_murmur_outcome_df)
    # embs_val, labels_val = load_embs_labels(val_embs_folder_murmur, "murmur", patient_murmur_outcome_df)
    logger.info("Loading sets for murmur decision...")
    #TODO: Shuffle embs_train WITH LABELS. Than BATCH!
    temp = list(zip(embs_train, embs_label_train))
    random.shuffle(temp)
    embs_train, embs_label_train = zip(*temp)
    
    temp = list(zip(embs_val, embs_label_val))
    random.shuffle(temp)
    embs_val, embs_label_val = zip(*temp)
    
    temp = list(zip(embs_test, embs_label_test))
    random.shuffle(temp)
    embs_test, embs_label_test = zip(*temp)
    
    
    train_decision_dataset = tf.data.Dataset.from_tensor_slices((np.vstack(embs_train), list(embs_label_train))).batch(min(len(embs_train), len(embs_val), ALGORITHM_HPS[BATCH_SIZE_DECISION_lbl]), drop_remainder=True)
    val_decision_dataset = tf.data.Dataset.from_tensor_slices((np.vstack(embs_val), list(embs_label_val))).batch(min(len(embs_train), len(embs_val), ALGORITHM_HPS[BATCH_SIZE_DECISION_lbl]))
    test_decision_dataset = tf.data.Dataset.from_tensor_slices((np.vstack(embs_test), list(embs_label_test))).batch(1)

    sklearn_weights_decision = class_weight.compute_class_weight("balanced",classes=[False, True], y= np.vstack(train_decision_dataset.map(lambda x,y: y)).flatten().tolist())
    sklearn_weights_decision = dict(enumerate(sklearn_weights_decision))
    sklearn_weights_decision[1] *= ALGORITHM_HPS[class_weight_decision_lbl]
    
    
    if FINAL_TRAINING:
        train_decision_dataset = train_decision_dataset.concatenate(val_decision_dataset)
        val_decision_dataset = test_decision_dataset
    
    if RUN_AUTOKERAS_DECISION:
        input = ak.Input(name=None)
        output_node = ak.DenseBlock(num_layers=None, num_units=None, dropout=None)(input)
        output_node = ak.ClassificationHead(dropout=0)(output_node)
        clf = ak.AutoModel(
            input,
            output_node,
            project_name="auto_model",
            max_trials=MAX_TRIALS,
            directory=None,
            tuner="bayesian",
            overwrite=True,
            seed=42,
            objective = kt.Objective("val_auc", direction="max"),
            max_model_size=None, 
            metrics = get_all_metrics()
        )
        clf.fit(train_decision_dataset, validation_data = val_decision_dataset, epochs = MURMUR_DECISION_EPOCHS, class_weight=sklearn_weights_decision, callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_compute_weighted_accuracy",
                min_delta=0,
                patience=10,
                verbose=0,
                mode="max",
                baseline=None,
                restore_best_weights=True
            )], workers= WORKERS)
        murmur_decision_new = keras.models.load_model(clf.tuner.best_model_path, custom_objects={"CustomLayer": CastToFloat32, "compute_weighted_accuracy": compute_weighted_accuracy })
        
    else:
        if USE_COMPLEX_MODELS:
            murmur_decision_new = get_murmur_decision_model() 
            # murmur_decision_new = get_murmur_decision_model_pretrained(murmur_model_new) 
            
        else:
            murmur_decision_config = get_murmur_decision_model_configs()
            murmur_decision_new = Functional.from_config(murmur_decision_config) 
            murmur_decision_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'learning_rate': 2e-05,'beta_1': 0.8999999761581421, 'beta_2': 0.9990000128746033, 'epsilon': 1e-07, 'amsgrad': False}), loss="binary_crossentropy", metrics=get_all_metrics())
        
        
        murmur_decision_new.fit(train_decision_dataset, max_queue_size=MAX_QUEUE, validation_data = val_decision_dataset, epochs = MURMUR_DECISION_EPOCHS, class_weight=sklearn_weights_decision, callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                min_delta=0,
                patience=20,
                verbose=0,
                mode="max",
                baseline=None,
                restore_best_weights=True
            )], workers=WORKERS)
        
        logger.info("Murmur Detection Model Classification Report")
        decision_evaluation = murmur_decision_new.evaluate(test_decision_dataset, return_dict=True) 
        logger.info(decision_evaluation)
        # Embs Size : [16, 64, 256]
        # Weight class murmur : [1, 1.5, 3, 5]
        # Weight class decision : [1, 1.5, 3, 5]
        # Murmur Image Size : [32, 72, 108, 216]
        # Random embeddings per patient: [15, 50, 100, 128]
        # Reshuffle for training: [1, 3, 5, 10]
        json_opject = json.dumps(decision_evaluation, indent=4)
        fptr = open('decision_evaluation.json', 'w')
        fptr.write(json_opject)
        fptr.close()
        with tf.summary.create_file_writer("logs_hparams/{}".format(get_unique_name())).as_default():
            hparams = {
                EMBS_SIZE_lbl : ALGORITHM_HPS[EMBS_SIZE_lbl],
                class_weight_murmur_lbl : ALGORITHM_HPS[class_weight_murmur_lbl],
                class_weight_decision_lbl : ALGORITHM_HPS[class_weight_decision_lbl],
                MURMUR_IMAGE_SIZE_lbl : ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl],
                EMBDS_PER_PATIENTS_lbl : ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl],
                RESHUFFLE_PATIENT_EMBS_N_lbl : ALGORITHM_HPS[RESHUFFLE_PATIENT_EMBS_N_lbl] 
            }
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar("auc", decision_evaluation["auc"], step=1)
            tf.summary.scalar("auc_roc", decision_evaluation["auc_roc"], step=1)
            tf.summary.scalar("compute_weighted_accuracy", decision_evaluation["compute_weighted_accuracy"], step=1)
        
    tf.keras.models.save_model(
                murmur_decision_new,
                os.path.join(model_folder, 'murmur_decision.tf'),
                overwrite=True,
                include_optimizer=True,
                save_format="tf",
                signatures=None,
                options=None,
                save_traces=True
            )
    # Train outcome model
    patients_file_informations_df["weight"] = patients_file_informations_df["weight"].fillna(patients_file_informations_df["weight"].mean())
    patients_file_informations_df["height"] = patients_file_informations_df["height"].fillna(patients_file_informations_df["height"].mean())
    patients_file_informations_df["age"] = patients_file_informations_df["age"].fillna(patients_file_informations_df["age"].mode()[0])
    patients_file_informations_df["age"] = patients_file_informations_df["age"].replace("nan", patients_file_informations_df["age"].mode()[0]).values
    patients_file_informations_df["bmi"] = patients_file_informations_df["weight"] / ((patients_file_informations_df["height"] / 100)**2)
    patients_file_informations_df["sex"] = (patients_file_informations_df["sex"] == "Male").astype(float)
    patients_file_informations_df["pregnancy_status"] = patients_file_informations_df["pregnancy_status"].astype(float)
    vocab_age = feature_column.categorical_column_with_vocabulary_list('Type', patients_file_informations_df["age"].unique())
    age_column_indicator = feature_column.indicator_column(vocab_age)
    patients_file_informations_df = pd.concat([patients_file_informations_df, pd.get_dummies(patients_file_informations_df["age"])], axis=1)
    demographics = ["pregnancy_status", "weight", "height", "sex", "bmi"] +  patients_file_informations_df["age"].unique().tolist()
    patients_file_informations_values_df = patients_file_informations_df[demographics]
    patients_file_informations_values_df = (patients_file_informations_values_df - patients_file_informations_values_df.mean()) / patients_file_informations_values_df.std()
    patients_file_informations_df[demographics] = patients_file_informations_values_df[demographics].values
    patients_file_informations_df["patient_id"] = patients_file_informations_df["file_prefix"].apply(lambda x: x.split("_")[0])
    patients_file_informations_df = patients_file_informations_df.drop_duplicates(subset=["patient_id"])
    embs_patient_id_df = pd.DataFrame(zip([*embs_train] + [*embs_val] + [*embs_test], embs_patient_id_train + embs_patient_id_val + embs_patient_id_test), columns=["embs", "patient_id"])
    
    #Prepare input dataset
    patients_file_informations_df = patients_file_informations_df.merge(embs_patient_id_df, left_on="patient_id", right_on="patient_id")
    inputs_features = demographics + ["embs"] 
    input_X = pd.concat([pd.DataFrame(np.vstack(patients_file_informations_df["embs"].values)), patients_file_informations_df[[*filter(lambda x: x != "embs", inputs_features)]]], axis=1)
    input_y = (patients_file_informations_df["outcome"] == "Abnormal").astype(float)
    train_idx = patients_file_informations_df[patients_file_informations_df["split"] == "train"].index
    val_idx = patients_file_informations_df[patients_file_informations_df["split"] == "val"].index
    test_idx = patients_file_informations_df[patients_file_informations_df["split"] == "test"].index
    train_input_X = input_X.loc[train_idx].values
    val_input_X = input_X.loc[val_idx].values
    test_input_X = input_X.loc[test_idx].values
    train_input_y = input_y.loc[train_idx].values
    val_input_y = input_y.loc[val_idx].values
    test_input_y = input_y.loc[test_idx].values
    
    # tf.config.run_functions_eagerly(False)
    # tuner = kt.tuners.RandomSearch(max_trials=20)
    # tuner.discret("max_depth", [4, 5, 6, 7])
    # model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
    # model.run_eagerly
    # import ipdb;ipdb.set_trace()
    # pass   
    
    # train_outcome_df_dataset = tf.data.Dataset.from_tensor_slices((np.vstack(train_input_X), train_input_y.reshape(-1,1)))
    
    # train_outcome_df = pd.concat([pd.DataFrame(train_input_X, columns=[*map(lambda x: str(x), pd.DataFrame(train_input_X).columns)]), pd.DataFrame(train_input_y, columns= ["label"])], axis=1)
    # train_outcome_df_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_outcome_df, label="label")
    # val_outcome_df = pd.concat([pd.DataFrame(val_input_X, columns=[*map(lambda x: str(x), pd.DataFrame(val_input_X).columns)]), pd.DataFrame(val_input_y, columns= ["label"])], axis=1)
    # val_outcome_df_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(val_outcome_df, label="label")
    # # test_outcome_df = pd.concat([pd.DataFrame(test_input_X, columns=[*map(lambda x: str(x), pd.DataFrame(test_input_X).columns)]), pd.DataFrame(test_input_y, columns= ["label"])], axis=1)
    # # test_outcome_df_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_outcome_df, label="label")
    # # model.fit(train_outcome_df_dataset)
    # input_layer = tf.keras.layers.InputLayer(input_shape=train_input_X.shape[1])
    # layer_1 = CastToFloat32.from_config({'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True})
    # # layer_dense = tf.keras.layers.Dense(8, activation="re_lu"),
    # # layer_dropout = tf.keras.layers.Dropout(0.5, seed=42),
    # model_layers = [input_layer, layer_1]
    # # for _ in range(ALGORITHM_HPS[N_DECISION_LAYERS_lbl]):
    # #     model_layers.append(tf.keras.layers.Dense(ALGORITHM_HPS[NEURONS_DECISION_lbl], activation="relu"))
    # #     if ALGORITHM_HPS[IS_DROPOUT_IN_DECISION_lbl]:
    # #         model_layers.append(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_DECISION_lbl], seed=42))
    # model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    # murmur_outcome_new = tf.keras.Sequential(model_layers)
    # murmur_outcome_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'decay':0.0, 'learning_rate': 0.0001,'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}), loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=get_all_metrics())
    
    # murmur_outcome_new.fit(train_outcome_df, validation_data=val_outcome_df)
    
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
    models_info = pd.read_pickle(os.path.join(model_folder,"models_info.pickle"))
    return {"noise_model" : noise_model, 
            "murmur_model" : murmur_model, 
            "murmur_decision_model": murmur_decision_model,
            EMBDS_PER_PATIENTS_lbl : ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl],
            EMBS_SIZE_lbl : models_info[EMBS_SIZE_lbl],
            NOISE_IMAGE_SIZE_lbl : models_info[NOISE_IMAGE_SIZE_lbl],
            MURMUR_IMAGE_SIZE_lbl : models_info[MURMUR_IMAGE_SIZE_lbl],
            }

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
        if verbose > 0:
            logger.info("File saved: {}".format(filename))
    clean_frequency_folder(AUX_FOLDER)
    clean_files = glob.glob(os.path.join(AUX_FOLDER, "*_clean.wav"))
    for clean_file in clean_files:
        generate_mel_wav_crops_v2((clean_file, AUX_IMGS_FOLDER))

    imgs_to_filter = tf.keras.utils.image_dataset_from_directory(AUX_IMGS_FOLDER, labels = None, image_size=(model[NOISE_IMAGE_SIZE_lbl], model[NOISE_IMAGE_SIZE_lbl]))
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
    murmur_embeddings_model = tf.keras.models.Sequential(model["murmur_model"].layers[:EMBEDDING_LAYER_REFERENCE_MURMUR_MODEL])
    imgs_to_emb = tf.keras.utils.image_dataset_from_directory(AUX_IMGS_FOLDER, labels = None, image_size=(model[MURMUR_IMAGE_SIZE_lbl], model[MURMUR_IMAGE_SIZE_lbl]))
    embs = murmur_embeddings_model.predict(imgs_to_emb)
    embs_df = pd.DataFrame(embs).sample(frac=1, random_state=42)

    if embs_df.shape[0] < model[EMBDS_PER_PATIENTS_lbl]:
        embs_df = embs_df.sample(model[EMBDS_PER_PATIENTS_lbl], replace=True, random_state=42)
    else:
        embs_df = embs_df.head(model[EMBDS_PER_PATIENTS_lbl])
    
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
    pd.Series({
        EMBDS_PER_PATIENTS_lbl : ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl],
        EMBS_SIZE_lbl : ALGORITHM_HPS[EMBS_SIZE_lbl],
        NOISE_IMAGE_SIZE_lbl : ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl],
        MURMUR_IMAGE_SIZE_lbl : ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl]
    }).to_pickle(os.path.join(model_folder, "models_info.pickle"))
    
    if OHH_ARGS and ("AWS_ID" in OHH_ARGS):
        destinypath = "/tmp/models-{}.tar.gz".format(str(uuid.uuid4()))
        os.system("tar -cvzf {} {}".format(destinypath, model_folder))
        response = s3.upload_file(destinypath, "1hh-algorithm-dev", "models/" + os.path.basename(destinypath))


def get_murmur_decision_model_pretrained(murmur_model):
    input_layer = tf.keras.layers.InputLayer(input_shape=(ALGORITHM_HPS[EMBS_SIZE_lbl] * ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl],))
    layer_1 = CastToFloat32.from_config({'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True})
    # layer_dense = tf.keras.layers.Dense(8, activation="re_lu"),
    # layer_dropout = tf.keras.layers.Dropout(0.5, seed=42),
    model_layers = [input_layer, layer_1]
    dense_layer = tf.keras.layers.Dense(ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl], activation="sigmoid")
    import ipdb;ipdb.set_trace()
    pass
    model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    murmur_decision_new = tf.keras.Sequential(model_layers)
    return murmur_decision_new

def get_murmur_decision_model():
    input_layer = tf.keras.layers.InputLayer(input_shape=(ALGORITHM_HPS[EMBS_SIZE_lbl] * ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl],))
    layer_1 = CastToFloat32.from_config({'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True})
    # layer_dense = tf.keras.layers.Dense(8, activation="re_lu"),
    # layer_dropout = tf.keras.layers.Dropout(0.5, seed=42),
    model_layers = [input_layer, layer_1]
    for _ in range(ALGORITHM_HPS[N_DECISION_LAYERS_lbl]):
        model_layers.append(tf.keras.layers.Dense(ALGORITHM_HPS[NEURONS_DECISION_lbl], activation="relu"))
        if ALGORITHM_HPS[IS_DROPOUT_IN_DECISION_lbl]:
            model_layers.append(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_DECISION_lbl], seed=42))
    model_layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    murmur_decision_new = tf.keras.Sequential(model_layers)
    murmur_decision_new.compile(optimizer=tf.keras.optimizers.Adam.from_config({'name': 'Adam', 'decay':0.0, 'learning_rate': 0.0001,'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}), loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=get_all_metrics())
    return murmur_decision_new
    
     
def get_murmur_decision_model_configs():
    murmur_decision_config = {'name': 'model', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, ALGORITHM_HPS[EMBS_SIZE_lbl] * ALGORITHM_HPS[EMBDS_PER_PATIENTS_lbl]), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}, 'name': 'input_1', 'inbound_nodes': []}, {'class_name': 'Custom>CastToFloat32', 'config': {'name': 'cast_to_float32', 'trainable': True, 'dtype': 'float32'}, 'name': 'cast_to_float32', 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 1024, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense', 'inbound_nodes': [[['cast_to_float32', 0, 0, {}]]]}, {'class_name': 'ReLU', 'config': {'name': 're_lu', 'trainable': True, 'dtype': 'float32', 'max_value': None, 'negative_slope': array(0., dtype=float32), 'threshold': array(0., dtype=float32)}, 'name': 're_lu', 'inbound_nodes': [[['dense', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_1', 'inbound_nodes': [[['re_lu', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'classification_head_1', 'trainable': True, 'dtype': 'float32', 'activation': 'sigmoid'}, 'name': 'classification_head_1', 'inbound_nodes': [[['dense_1', 0, 0, {}]]]}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['classification_head_1', 0, 0]]}
    return murmur_decision_config

def get_murmur_model():
    # return ak.ImageClassifier()
    if ALGORITHM_HPS[IS_MURMUR_MODEL_XCEPTION_lbl]:
        noise_input_config = {'batch_input_shape': (None, ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], 3),  'dtype': 'float32',  'name': 'input_1', 'ragged': False, 'sparse': False }
        noise_cast_to_float_config = {'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True}

        noise_global_average_config = {'data_format': 'channels_last', 'dtype': 'float32', 'keepdims': False, 'name': 'global_average_pooling2d', 'trainable': True}
        noise_dense_config = {'activation': 'sigmoid',  'activity_regularizer': None,  'bias_constraint': None, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'bias_regularizer': None,  'dtype': 'float32', 'kernel_constraint': None, 'kernel_initializer': {'class_name': 'GlorotUniform',  'config': {'seed': 42}}, 'kernel_regularizer': None, 'name': 'dense', 'trainable': True, 'units': 1, 'use_bias': True}
        # noise_classification_head_config = {'activation': 'sigmoid', 'dtype': 'float32', 'name': 'classification_head_1', 'trainable': True}
        
        noise_layer_0 = tf.keras.Input(**noise_input_config)
        noise_layer_1 = CastToFloat32.from_config(noise_cast_to_float_config)
        noise_layer_3 = tf.keras.applications.xception.Xception(
                        include_top=False,
                        input_tensor=None,
                        weights="imagenet",
                        # pooling="avg",
                        classes=2,
                        classifier_activation='None'
                        )
        noise_layer_4 = tf.keras.layers.GlobalAveragePooling2D.from_config(noise_global_average_config)
        noise_layer_embs = tf.keras.layers.Dense(ALGORITHM_HPS[EMBS_SIZE_lbl], activation="linear",name="dense_embs")
        noise_layer_5 = tf.keras.layers.Dense.from_config(noise_dense_config)
        # noise_layer_6 = tf.keras.layers.Activation.from_config(noise_classification_head_config)
        
        murmur_model = tf.keras.Sequential([noise_layer_0, noise_layer_1, tf.keras.layers.Rescaling(1./127.5), noise_layer_3,  
        noise_layer_4, noise_layer_embs, noise_layer_5])
    else:
        murmur_model = tf.keras.models.Sequential()
        murmur_model.add(tf.keras.layers.Conv2D(ALGORITHM_HPS[N_MURMUR_CNN_NEURONS_LAYERS_lbl], (3, 3), activation='relu', input_shape=(ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], 3)))
        murmur_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        if ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]:
            murmur_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
        for _ in range(ALGORITHM_HPS[N_MURMUR_LAYERS_lbl]):
            murmur_model.add(tf.keras.layers.Conv2D(ALGORITHM_HPS[N_MURMUR_CNN_NEURONS_LAYERS_lbl], (3, 3), activation='relu'))
            murmur_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            if ALGORITHM_HPS[IS_DROPOUT_IN_MURMUR_lbl]:
                murmur_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
        murmur_model.add(tf.keras.layers.Flatten())
        if ALGORITHM_HPS[IS_DROPOUT_IN_MURMUR_lbl]:
                murmur_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
        murmur_model.add(tf.keras.layers.Dense(ALGORITHM_HPS[EMBS_SIZE_lbl], activation='relu'))
        murmur_model.add(tf.keras.layers.BatchNormalization())
        murmur_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                
    optimizer = tf.keras.optimizers.Adam(
                       learning_rate=0.0001
                    )  
    murmur_model.compile(optimizer=optimizer, metrics=get_all_metrics(), loss="binary_crossentropy",)
    return murmur_model

def get_murmur_model_configs(): 
    murmur_config = {'name': 'model', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], ALGORITHM_HPS[MURMUR_IMAGE_SIZE_lbl], 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_1'}, 'name': 'input_1', 'inbound_nodes': []}, {'class_name': 'Custom>CastToFloat32', 'config': {'name': 'cast_to_float32', 'trainable': True, 'dtype': 'float32'}, 'name': 'cast_to_float32', 'inbound_nodes': [[['input_1', 0, 0, {}]]]}, {'class_name': 'Resizing', 'config': {'name': 'resizing', 'trainable': True, 'dtype': 'float32', 'height': 224, 'width': 224, 'interpolation': 'bilinear', 'crop_to_aspect_ratio': False}, 'name': 'resizing', 'inbound_nodes': [[['cast_to_float32', 0, 0, {}]]]}, {'class_name': 'Functional', 'config': {'name': 'xception', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': (None, 224, 224, 3), 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'input_2'}, 'name': 'input_2', 'inbound_nodes': []}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv1', 'trainable': True, 'dtype': 'float32', 'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv1', 'inbound_nodes': [[['input_2', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv1_bn', 'trainable': True, 'dtype': 'float32', 'axis':[3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv1_bn', 'inbound_nodes': [[['block1_conv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv1_act', 'inbound_nodes': [[['block1_conv1_bn', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'block1_conv2', 'trainable': True, 'dtype': 'float32', 'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'block1_conv2', 'inbound_nodes': [[['block1_conv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block1_conv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block1_conv2_bn', 'inbound_nodes': [[['block1_conv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block1_conv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block1_conv2_act', 'inbound_nodes': [[['block1_conv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 24}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv1', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv1_bn', 'inbound_nodes': [[['block2_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block2_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block2_sepconv2_act', 'inbound_nodes': [[['block2_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block2_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 35}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block2_sepconv2', 'inbound_nodes': [[['block2_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block2_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block2_sepconv2_bn', 'inbound_nodes': [[['block2_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d', 'trainable': True, 'dtype': 'float32', 'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d', 'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block2_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block2_pool', 'inbound_nodes': [[['block2_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization', 'inbound_nodes': [[['conv2d', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add', 'trainable': True, 'dtype': 'float32'}, 'name': 'add', 'inbound_nodes': [[['block2_pool', 0, 0, {}], ['batch_normalization', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv1_act', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 56}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv1', 'inbound_nodes': [[['block3_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv1_bn', 'inbound_nodes': [[['block3_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block3_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block3_sepconv2_act', 'inbound_nodes': [[['block3_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block3_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 67}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block3_sepconv2', 'inbound_nodes': [[['block3_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block3_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block3_sepconv2_bn', 'inbound_nodes': [[['block3_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_1', 'trainable': True, 'dtype': 'float32', 'filters': 256, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_1', 'inbound_nodes': [[['add', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block3_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block3_pool', 'inbound_nodes': [[['block3_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_1', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_1', 'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_1', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_1', 'inbound_nodes': [[['block3_pool', 0, 0, {}], ['batch_normalization_1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv1_act', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 88}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv1', 'inbound_nodes': [[['block4_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv1_bn', 'inbound_nodes': [[['block4_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block4_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block4_sepconv2_act', 'inbound_nodes': [[['block4_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block4_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 99}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block4_sepconv2', 'inbound_nodes': [[['block4_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block4_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block4_sepconv2_bn', 'inbound_nodes': [[['block4_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_2', 'inbound_nodes': [[['add_1', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block4_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block4_pool', 'inbound_nodes': [[['block4_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_2', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_2', 'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_2', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_2', 'inbound_nodes': [[['block4_pool', 0, 0, {}], ['batch_normalization_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv1_act', 'inbound_nodes': [[['add_2', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 120}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv1', 'inbound_nodes': [[['block5_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv1_bn', 'inbound_nodes': [[['block5_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv2_act', 'inbound_nodes': [[['block5_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 131}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv2', 'inbound_nodes': [[['block5_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv2_bn', 'inbound_nodes': [[['block5_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block5_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block5_sepconv3_act', 'inbound_nodes': [[['block5_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block5_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 142}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block5_sepconv3', 'inbound_nodes': [[['block5_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block5_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block5_sepconv3_bn', 'inbound_nodes': [[['block5_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_3', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_3', 'inbound_nodes': [[['block5_sepconv3_bn', 0, 0, {}], ['add_2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv1_act', 'inbound_nodes': [[['add_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 154}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv1', 'inbound_nodes': [[['block6_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv1_bn', 'inbound_nodes': [[['block6_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv2_act', 'inbound_nodes': [[['block6_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 165}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv2', 'inbound_nodes': [[['block6_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv2_bn', 'inbound_nodes': [[['block6_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block6_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block6_sepconv3_act', 'inbound_nodes': [[['block6_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block6_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 176}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block6_sepconv3', 'inbound_nodes': [[['block6_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block6_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block6_sepconv3_bn', 'inbound_nodes': [[['block6_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_4', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_4', 'inbound_nodes': [[['block6_sepconv3_bn', 0, 0, {}], ['add_3', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv1_act', 'inbound_nodes': [[['add_4', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 188}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv1', 'inbound_nodes': [[['block7_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv1_bn', 'inbound_nodes': [[['block7_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv2_act', 'inbound_nodes': [[['block7_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 199}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv2', 'inbound_nodes': [[['block7_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv2_bn', 'inbound_nodes': [[['block7_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block7_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block7_sepconv3_act', 'inbound_nodes': [[['block7_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block7_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 210}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block7_sepconv3', 'inbound_nodes': [[['block7_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block7_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block7_sepconv3_bn', 'inbound_nodes': [[['block7_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_5', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_5', 'inbound_nodes': [[['block7_sepconv3_bn', 0, 0, {}], ['add_4', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv1_act', 'inbound_nodes': [[['add_5', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 222}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv1', 'inbound_nodes': [[['block8_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv1_bn', 'inbound_nodes': [[['block8_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv2_act', 'inbound_nodes': [[['block8_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 233}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv2', 'inbound_nodes': [[['block8_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv2_bn', 'inbound_nodes': [[['block8_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block8_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block8_sepconv3_act', 'inbound_nodes': [[['block8_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block8_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 244}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block8_sepconv3', 'inbound_nodes': [[['block8_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block8_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block8_sepconv3_bn', 'inbound_nodes': [[['block8_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_6', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_6', 'inbound_nodes': [[['block8_sepconv3_bn', 0, 0, {}], ['add_5', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv1_act', 'inbound_nodes': [[['add_6', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 256}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv1', 'inbound_nodes': [[['block9_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv1_bn', 'inbound_nodes': [[['block9_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv2_act', 'inbound_nodes': [[['block9_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 267}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv2', 'inbound_nodes': [[['block9_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv2_bn', 'inbound_nodes': [[['block9_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block9_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block9_sepconv3_act', 'inbound_nodes': [[['block9_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block9_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 278}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block9_sepconv3', 'inbound_nodes': [[['block9_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block9_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block9_sepconv3_bn', 'inbound_nodes': [[['block9_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_7', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_7', 'inbound_nodes': [[['block9_sepconv3_bn', 0, 0, {}], ['add_6', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv1_act', 'inbound_nodes': [[['add_7', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 290}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv1', 'inbound_nodes': [[['block10_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv1_bn', 'inbound_nodes': [[['block10_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv2_act', 'inbound_nodes': [[['block10_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 301}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv2', 'inbound_nodes': [[['block10_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv2_bn', 'inbound_nodes': [[['block10_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block10_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block10_sepconv3_act', 'inbound_nodes': [[['block10_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block10_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 312}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block10_sepconv3', 'inbound_nodes': [[['block10_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block10_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block10_sepconv3_bn', 'inbound_nodes': [[['block10_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_8', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_8', 'inbound_nodes': [[['block10_sepconv3_bn', 0, 0, {}], ['add_7', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv1_act', 'inbound_nodes': [[['add_8', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 324}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv1', 'inbound_nodes': [[['block11_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv1_bn', 'inbound_nodes': [[['block11_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv2_act', 'inbound_nodes': [[['block11_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 335}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv2', 'inbound_nodes': [[['block11_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv2_bn', 'inbound_nodes': [[['block11_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block11_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block11_sepconv3_act', 'inbound_nodes': [[['block11_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block11_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 346}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block11_sepconv3', 'inbound_nodes': [[['block11_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block11_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block11_sepconv3_bn', 'inbound_nodes': [[['block11_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_9', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_9', 'inbound_nodes': [[['block11_sepconv3_bn', 0, 0, {}], ['add_8', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv1_act', 'inbound_nodes': [[['add_9', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 358}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv1', 'inbound_nodes': [[['block12_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv1_bn', 'inbound_nodes': [[['block12_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv2_act', 'inbound_nodes': [[['block12_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 369}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv2', 'inbound_nodes': [[['block12_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv2_bn', 'inbound_nodes': [[['block12_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block12_sepconv3_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block12_sepconv3_act', 'inbound_nodes': [[['block12_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block12_sepconv3', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 380}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block12_sepconv3', 'inbound_nodes': [[['block12_sepconv3_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block12_sepconv3_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block12_sepconv3_bn', 'inbound_nodes': [[['block12_sepconv3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_10', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_10', 'inbound_nodes': [[['block12_sepconv3_bn', 0, 0, {}], ['add_9', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv1_act', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 392}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv1', 'inbound_nodes': [[['block13_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv1_bn', 'inbound_nodes': [[['block13_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block13_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block13_sepconv2_act', 'inbound_nodes': [[['block13_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block13_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 403}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block13_sepconv2', 'inbound_nodes': [[['block13_sepconv2_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block13_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block13_sepconv2_bn', 'inbound_nodes': [[['block13_sepconv2', 0, 0, {}]]]}, {'class_name': 'Conv2D', 'config': {'name': 'conv2d_3', 'trainable': True, 'dtype': 'float32', 'filters': 1024, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'conv2d_3', 'inbound_nodes': [[['add_10', 0, 0, {}]]]}, {'class_name': 'MaxPooling2D', 'config': {'name': 'block13_pool', 'trainable': True, 'dtype': 'float32', 'pool_size': (3, 3), 'padding': 'same', 'strides': (2, 2), 'data_format': 'channels_last'}, 'name': 'block13_pool', 'inbound_nodes': [[['block13_sepconv2_bn', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_3', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_3', 'inbound_nodes': [[['conv2d_3', 0, 0, {}]]]}, {'class_name': 'Add', 'config': {'name': 'add_11', 'trainable': True, 'dtype': 'float32'}, 'name': 'add_11', 'inbound_nodes': [[['block13_pool', 0, 0, {}], ['batch_normalization_3', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv1', 'trainable': True, 'dtype': 'float32', 'filters': 1536, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 423}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv1', 'inbound_nodes': [[['add_11', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv1_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv1_bn', 'inbound_nodes': [[['block14_sepconv1', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv1_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv1_act', 'inbound_nodes': [[['block14_sepconv1_bn', 0, 0, {}]]]}, {'class_name': 'SeparableConv2D', 'config': {'name': 'block14_sepconv2', 'trainable': True, 'dtype': 'float32', 'filters': 2048, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'activation': 'linear', 'use_bias': False, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}, 'shared_object_id': 434}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'depth_multiplier': 1, 'depthwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'pointwise_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'depthwise_regularizer': None, 'pointwise_regularizer': None, 'depthwise_constraint': None, 'pointwise_constraint': None}, 'name': 'block14_sepconv2', 'inbound_nodes': [[['block14_sepconv1_act', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'block14_sepconv2_bn', 'trainable': True, 'dtype': 'float32', 'axis': [3], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'block14_sepconv2_bn', 'inbound_nodes': [[['block14_sepconv2', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'block14_sepconv2_act', 'trainable': True, 'dtype': 'float32', 'activation': 'relu'}, 'name': 'block14_sepconv2_act', 'inbound_nodes': [[['block14_sepconv2_bn', 0, 0, {}]]]}], 'input_layers': [['input_2', 0, 0]], 'output_layers': [['block14_sepconv2_act', 0, 0]]}, 'name': 'xception', 'inbound_nodes': [[['resizing', 0, 0, {}]]]}, {'class_name': 'Flatten', 'config': {'name': 'flatten', 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}, 'name': 'flatten', 'inbound_nodes': [[['xception', 1, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense', 'trainable': True, 'dtype': 'float32', 'units': 64, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense', 'inbound_nodes': [[['flatten', 0, 0, {}]]]}, {'class_name': 'BatchNormalization', 'config': {'name': 'batch_normalization_4', 'trainable': True, 'dtype': 'float32', 'axis': [1], 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': {'class_name': 'Zeros', 'config': {}}, 'gamma_initializer': {'class_name': 'Ones', 'config': {}}, 'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}}, 'moving_variance_initializer': {'class_name': 'Ones', 'config': {}}, 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}, 'name': 'batch_normalization_4', 'inbound_nodes': [[['dense', 0, 0, {}]]]}, {'class_name': 'ReLU', 'config': {'name': 're_lu', 'trainable': True, 'dtype': 'float32', 'max_value': None, 'negative_slope': array(0., dtype=float32), 'threshold': array(0., dtype=float32)}, 'name': 're_lu', 'inbound_nodes': [[['batch_normalization_4', 0, 0, {}]]]}, {'class_name': 'Dense', 'config': {'name': 'dense_1', 'trainable': True, 'dtype': 'float32', 'units': 1, 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': 42}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}, 'name': 'dense_1', 'inbound_nodes': [[['re_lu', 0, 0, {}]]]}, {'class_name': 'Activation', 'config': {'name': 'classification_head_1', 'trainable': True, 'dtype': 'float32', 'activation': 'sigmoid'}, 'name': 'classification_head_1', 'inbound_nodes': [[['dense_1', 0, 0, {}]]]}], 'input_layers': [['input_1', 0, 0]], 'output_layers': [['classification_head_1', 0, 0]]}
    return murmur_config


def get_noise_model_v2(): 
    noise_model = tf.keras.models.Sequential()
    noise_model.add(tf.keras.layers.Conv2D(ALGORITHM_HPS[N_MURMUR_CNN_NEURONS_LAYERS_lbl], (3, 3), activation='relu', input_shape=(ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], 3)))
    noise_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]:
        noise_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
    for _ in range(ALGORITHM_HPS[N_MURMUR_LAYERS_lbl]):
        noise_model.add(tf.keras.layers.Conv2D(ALGORITHM_HPS[N_MURMUR_CNN_NEURONS_LAYERS_lbl], (3, 3), activation='relu'))
        noise_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        if ALGORITHM_HPS[IS_DROPOUT_IN_MURMUR_lbl]:
            noise_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
    noise_model.add(tf.keras.layers.Flatten())
    if ALGORITHM_HPS[IS_DROPOUT_IN_MURMUR_lbl]:
            noise_model.add(tf.keras.layers.Dropout(ALGORITHM_HPS[DROPOUT_VALUE_IN_MURMUR_lbl]))
    noise_model.add(tf.keras.layers.Dense(ALGORITHM_HPS[EMBS_SIZE_lbl], activation='relu'))
    noise_model.add(tf.keras.layers.BatchNormalization())
    noise_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                
    optimizer = tf.keras.optimizers.Adam(
                       learning_rate=0.0001
                    )  
    noise_model.compile(optimizer=optimizer, metrics=get_all_metrics(), loss="binary_crossentropy",)
    
    # noise_input_config = {'batch_input_shape': (None, ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], NOISE_IMAGE_SIZE, 3),  'dtype': 'float32',  'name': 'input_1', 'ragged': False, 'sparse': False }
    # noise_cast_to_float_config = {'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True}

    # noise_global_average_config = {'data_format': 'channels_last', 'dtype': 'float32', 'keepdims': False, 'name': 'global_average_pooling2d', 'trainable': True}
    # noise_dense_config = {'activation': 'linear',  'activity_regularizer': None,  'bias_constraint': None, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'bias_regularizer': None,  'dtype': 'float32', 'kernel_constraint': None, 'kernel_initializer': {'class_name': 'GlorotUniform',  'config': {'seed': 42}}, 'kernel_regularizer': None, 'name': 'dense', 'trainable': True, 'units': 1, 'use_bias': True}
    # noise_classification_head_config = {'activation': 'sigmoid', 'dtype': 'float32', 'name': 'classification_head_1', 'trainable': True}
    
    # noise_layer_0 = tf.keras.Input(**noise_input_config)
    # noise_layer_1 = CastToFloat32.from_config(noise_cast_to_float_config)
    # noise_layer_3 = tf.keras.applications.xception.Xception(
    #                 include_top=False,
    #                 input_tensor=None,
    #                 weights="imagenet",
    #                 # pooling="avg",
    #                 classes=2,
    #                 classifier_activation='None'
    #                 )
    # noise_layer_4 = tf.keras.layers.GlobalAveragePooling2D.from_config(noise_global_average_config)
    # noise_layer_5 = tf.keras.layers.Dense.from_config(noise_dense_config)
    # noise_layer_6 = tf.keras.layers.Activation.from_config(noise_classification_head_config)
    
    # noise_model = tf.keras.Sequential([noise_layer_0, noise_layer_1, tf.keras.layers.Rescaling(1./127.5), noise_layer_3,  
    # noise_layer_4, noise_layer_5, noise_layer_6])
   
    
    return noise_model

def get_noise_model(): 
    noise_input_config = {'batch_input_shape': (None, ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], 3),  'dtype': 'float32',  'name': 'input_1', 'ragged': False, 'sparse': False }
    noise_cast_to_float_config = {'dtype': 'float32', 'name': 'cast_to_float32', 'trainable': True}

    noise_global_average_config = {'data_format': 'channels_last', 'dtype': 'float32', 'keepdims': False, 'name': 'global_average_pooling2d', 'trainable': True}
    noise_dense_config = {'activation': 'linear',  'activity_regularizer': None,  'bias_constraint': None, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'bias_regularizer': None,  'dtype': 'float32', 'kernel_constraint': None, 'kernel_initializer': {'class_name': 'GlorotUniform',  'config': {'seed': 42}}, 'kernel_regularizer': None, 'name': 'dense', 'trainable': True, 'units': 1, 'use_bias': True}
    noise_classification_head_config = {'activation': 'sigmoid', 'dtype': 'float32', 'name': 'classification_head_1', 'trainable': True}
    
    noise_layer_0 = tf.keras.Input(**noise_input_config)
    noise_layer_1 = CastToFloat32.from_config(noise_cast_to_float_config)
    noise_layer_3 = tf.keras.applications.xception.Xception(
                    include_top=False,
                    input_tensor=None,
                    weights="imagenet",
                    # pooling="avg",
                    classes=2,
                    classifier_activation='None'
                    )
    noise_layer_4 = tf.keras.layers.GlobalAveragePooling2D.from_config(noise_global_average_config)
    noise_layer_5 = tf.keras.layers.Dense.from_config(noise_dense_config)
    noise_layer_6 = tf.keras.layers.Activation.from_config(noise_classification_head_config)
    
    noise_model = tf.keras.Sequential([noise_layer_0, noise_layer_1, tf.keras.layers.Rescaling(1./127.5), noise_layer_3,  
    noise_layer_4, noise_layer_5, noise_layer_6])
   
    
    return noise_model

def get_noise_model_configs(): 
    noise_input_config = {'input_layers': [['input_1', 0, 0]],
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], ALGORITHM_HPS[NOISE_IMAGE_SIZE_lbl], 3),
    'dtype': 'float32',
    'name': 'input_1',
    'ragged': False,
    'sparse': False},
   'inbound_nodes': [],
   'name': 'input_1'},
  {'class_name': 'Custom>CastToFloat32',
   'config': {'dtype': 'float32',
    'name': 'cast_to_float32',
    'trainable': True},
   'inbound_nodes': [[['input_1', 0, 0, {}]]],
   'name': 'cast_to_float32'},
  {'class_name': 'Functional',
   'config': {'input_layers': [['input_2', 0, 0]],
    'layers': [{'class_name': 'InputLayer',
      'config': {'batch_input_shape': (None, None, None, 3),
       'dtype': 'float32',
       'name': 'input_2',
       'ragged': False,
       'sparse': False},
      'inbound_nodes': [],
      'name': 'input_2'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 32,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block1_conv1',
       'padding': 'valid',
       'strides': (2, 2),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['input_2', 0, 0, {}]]],
      'name': 'block1_conv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block1_conv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block1_conv1', 0, 0, {}]]],
      'name': 'block1_conv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block1_conv1_act',
       'trainable': True},
      'inbound_nodes': [[['block1_conv1_bn', 0, 0, {}]]],
      'name': 'block1_conv1_act'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 64,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block1_conv2',
       'padding': 'valid',
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block1_conv1_act', 0, 0, {}]]],
      'name': 'block1_conv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block1_conv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block1_conv2', 0, 0, {}]]],
      'name': 'block1_conv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block1_conv2_act',
       'trainable': True},
      'inbound_nodes': [[['block1_conv2_bn', 0, 0, {}]]],
      'name': 'block1_conv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 23},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 128,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block2_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]],
      'name': 'block2_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block2_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block2_sepconv1', 0, 0, {}]]],
      'name': 'block2_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block2_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block2_sepconv1_bn', 0, 0, {}]]],
      'name': 'block2_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 34},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 128,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block2_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block2_sepconv2_act', 0, 0, {}]]],
      'name': 'block2_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block2_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block2_sepconv2', 0, 0, {}]]],
      'name': 'block2_sepconv2_bn'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 128,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (1, 1),
       'name': 'conv2d',
       'padding': 'same',
       'strides': (2, 2),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block1_conv2_act', 0, 0, {}]]],
      'name': 'conv2d'},
     {'class_name': 'MaxPooling2D',
      'config': {'data_format': 'channels_last',
       'dtype': 'float32',
       'name': 'block2_pool',
       'padding': 'same',
       'pool_size': (3, 3),
       'strides': (2, 2),
       'trainable': True},
      'inbound_nodes': [[['block2_sepconv2_bn', 0, 0, {}]]],
      'name': 'block2_pool'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'batch_normalization',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['conv2d', 0, 0, {}]]],
      'name': 'batch_normalization'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add', 'trainable': True},
      'inbound_nodes': [[['block2_pool', 0, 0, {}],
        ['batch_normalization', 0, 0, {}]]],
      'name': 'add'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block3_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add', 0, 0, {}]]],
      'name': 'block3_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 55},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 256,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block3_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block3_sepconv1_act', 0, 0, {}]]],
      'name': 'block3_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block3_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block3_sepconv1', 0, 0, {}]]],
      'name': 'block3_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block3_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block3_sepconv1_bn', 0, 0, {}]]],
      'name': 'block3_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 66},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 256,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block3_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block3_sepconv2_act', 0, 0, {}]]],
      'name': 'block3_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block3_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block3_sepconv2', 0, 0, {}]]],
      'name': 'block3_sepconv2_bn'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 256,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (1, 1),
       'name': 'conv2d_1',
       'padding': 'same',
       'strides': (2, 2),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['add', 0, 0, {}]]],
      'name': 'conv2d_1'},
     {'class_name': 'MaxPooling2D',
      'config': {'data_format': 'channels_last',
       'dtype': 'float32',
       'name': 'block3_pool',
       'padding': 'same',
       'pool_size': (3, 3),
       'strides': (2, 2),
       'trainable': True},
      'inbound_nodes': [[['block3_sepconv2_bn', 0, 0, {}]]],
      'name': 'block3_pool'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'batch_normalization_1',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['conv2d_1', 0, 0, {}]]],
      'name': 'batch_normalization_1'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_1', 'trainable': True},
      'inbound_nodes': [[['block3_pool', 0, 0, {}],
        ['batch_normalization_1', 0, 0, {}]]],
      'name': 'add_1'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block4_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_1', 0, 0, {}]]],
      'name': 'block4_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 87},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block4_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block4_sepconv1_act', 0, 0, {}]]],
      'name': 'block4_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block4_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block4_sepconv1', 0, 0, {}]]],
      'name': 'block4_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block4_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block4_sepconv1_bn', 0, 0, {}]]],
      'name': 'block4_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 98},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block4_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block4_sepconv2_act', 0, 0, {}]]],
      'name': 'block4_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block4_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block4_sepconv2', 0, 0, {}]]],
      'name': 'block4_sepconv2_bn'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (1, 1),
       'name': 'conv2d_2',
       'padding': 'same',
       'strides': (2, 2),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['add_1', 0, 0, {}]]],
      'name': 'conv2d_2'},
     {'class_name': 'MaxPooling2D',
      'config': {'data_format': 'channels_last',
       'dtype': 'float32',
       'name': 'block4_pool',
       'padding': 'same',
       'pool_size': (3, 3),
       'strides': (2, 2),
       'trainable': True},
      'inbound_nodes': [[['block4_sepconv2_bn', 0, 0, {}]]],
      'name': 'block4_pool'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'batch_normalization_2',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['conv2d_2', 0, 0, {}]]],
      'name': 'batch_normalization_2'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_2', 'trainable': True},
      'inbound_nodes': [[['block4_pool', 0, 0, {}],
        ['batch_normalization_2', 0, 0, {}]]],
      'name': 'add_2'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block5_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_2', 0, 0, {}]]],
      'name': 'block5_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 119},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block5_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block5_sepconv1_act', 0, 0, {}]]],
      'name': 'block5_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block5_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block5_sepconv1', 0, 0, {}]]],
      'name': 'block5_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block5_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block5_sepconv1_bn', 0, 0, {}]]],
      'name': 'block5_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 130},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block5_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block5_sepconv2_act', 0, 0, {}]]],
      'name': 'block5_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block5_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block5_sepconv2', 0, 0, {}]]],
      'name': 'block5_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block5_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block5_sepconv2_bn', 0, 0, {}]]],
      'name': 'block5_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 141},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block5_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block5_sepconv3_act', 0, 0, {}]]],
      'name': 'block5_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block5_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block5_sepconv3', 0, 0, {}]]],
      'name': 'block5_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_3', 'trainable': True},
      'inbound_nodes': [[['block5_sepconv3_bn', 0, 0, {}],
        ['add_2', 0, 0, {}]]],
      'name': 'add_3'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block6_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_3', 0, 0, {}]]],
      'name': 'block6_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 153},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block6_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block6_sepconv1_act', 0, 0, {}]]],
      'name': 'block6_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block6_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block6_sepconv1', 0, 0, {}]]],
      'name': 'block6_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block6_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block6_sepconv1_bn', 0, 0, {}]]],
      'name': 'block6_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 164},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block6_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block6_sepconv2_act', 0, 0, {}]]],
      'name': 'block6_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block6_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block6_sepconv2', 0, 0, {}]]],
      'name': 'block6_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block6_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block6_sepconv2_bn', 0, 0, {}]]],
      'name': 'block6_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 175},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block6_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block6_sepconv3_act', 0, 0, {}]]],
      'name': 'block6_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block6_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block6_sepconv3', 0, 0, {}]]],
      'name': 'block6_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_4', 'trainable': True},
      'inbound_nodes': [[['block6_sepconv3_bn', 0, 0, {}],
        ['add_3', 0, 0, {}]]],
      'name': 'add_4'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block7_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_4', 0, 0, {}]]],
      'name': 'block7_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 187},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block7_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block7_sepconv1_act', 0, 0, {}]]],
      'name': 'block7_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block7_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block7_sepconv1', 0, 0, {}]]],
      'name': 'block7_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block7_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block7_sepconv1_bn', 0, 0, {}]]],
      'name': 'block7_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 198},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block7_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block7_sepconv2_act', 0, 0, {}]]],
      'name': 'block7_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block7_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block7_sepconv2', 0, 0, {}]]],
      'name': 'block7_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block7_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block7_sepconv2_bn', 0, 0, {}]]],
      'name': 'block7_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 209},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block7_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block7_sepconv3_act', 0, 0, {}]]],
      'name': 'block7_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block7_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block7_sepconv3', 0, 0, {}]]],
      'name': 'block7_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_5', 'trainable': True},
      'inbound_nodes': [[['block7_sepconv3_bn', 0, 0, {}],
        ['add_4', 0, 0, {}]]],
      'name': 'add_5'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block8_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_5', 0, 0, {}]]],
      'name': 'block8_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 221},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block8_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block8_sepconv1_act', 0, 0, {}]]],
      'name': 'block8_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block8_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block8_sepconv1', 0, 0, {}]]],
      'name': 'block8_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block8_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block8_sepconv1_bn', 0, 0, {}]]],
      'name': 'block8_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 232},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block8_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block8_sepconv2_act', 0, 0, {}]]],
      'name': 'block8_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block8_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block8_sepconv2', 0, 0, {}]]],
      'name': 'block8_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block8_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block8_sepconv2_bn', 0, 0, {}]]],
      'name': 'block8_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 243},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block8_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block8_sepconv3_act', 0, 0, {}]]],
      'name': 'block8_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block8_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block8_sepconv3', 0, 0, {}]]],
      'name': 'block8_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_6', 'trainable': True},
      'inbound_nodes': [[['block8_sepconv3_bn', 0, 0, {}],
        ['add_5', 0, 0, {}]]],
      'name': 'add_6'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block9_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_6', 0, 0, {}]]],
      'name': 'block9_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 255},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block9_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block9_sepconv1_act', 0, 0, {}]]],
      'name': 'block9_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block9_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block9_sepconv1', 0, 0, {}]]],
      'name': 'block9_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block9_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block9_sepconv1_bn', 0, 0, {}]]],
      'name': 'block9_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 266},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block9_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block9_sepconv2_act', 0, 0, {}]]],
      'name': 'block9_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block9_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block9_sepconv2', 0, 0, {}]]],
      'name': 'block9_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block9_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block9_sepconv2_bn', 0, 0, {}]]],
      'name': 'block9_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 277},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block9_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block9_sepconv3_act', 0, 0, {}]]],
      'name': 'block9_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block9_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block9_sepconv3', 0, 0, {}]]],
      'name': 'block9_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_7', 'trainable': True},
      'inbound_nodes': [[['block9_sepconv3_bn', 0, 0, {}],
        ['add_6', 0, 0, {}]]],
      'name': 'add_7'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block10_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_7', 0, 0, {}]]],
      'name': 'block10_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 289},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block10_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block10_sepconv1_act', 0, 0, {}]]],
      'name': 'block10_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block10_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block10_sepconv1', 0, 0, {}]]],
      'name': 'block10_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block10_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block10_sepconv1_bn', 0, 0, {}]]],
      'name': 'block10_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 300},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block10_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block10_sepconv2_act', 0, 0, {}]]],
      'name': 'block10_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block10_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block10_sepconv2', 0, 0, {}]]],
      'name': 'block10_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block10_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block10_sepconv2_bn', 0, 0, {}]]],
      'name': 'block10_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 311},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block10_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block10_sepconv3_act', 0, 0, {}]]],
      'name': 'block10_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block10_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block10_sepconv3', 0, 0, {}]]],
      'name': 'block10_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_8', 'trainable': True},
      'inbound_nodes': [[['block10_sepconv3_bn', 0, 0, {}],
        ['add_7', 0, 0, {}]]],
      'name': 'add_8'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block11_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_8', 0, 0, {}]]],
      'name': 'block11_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 323},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block11_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block11_sepconv1_act', 0, 0, {}]]],
      'name': 'block11_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block11_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block11_sepconv1', 0, 0, {}]]],
      'name': 'block11_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block11_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block11_sepconv1_bn', 0, 0, {}]]],
      'name': 'block11_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 334},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block11_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block11_sepconv2_act', 0, 0, {}]]],
      'name': 'block11_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block11_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block11_sepconv2', 0, 0, {}]]],
      'name': 'block11_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block11_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block11_sepconv2_bn', 0, 0, {}]]],
      'name': 'block11_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 345},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block11_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block11_sepconv3_act', 0, 0, {}]]],
      'name': 'block11_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block11_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block11_sepconv3', 0, 0, {}]]],
      'name': 'block11_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_9', 'trainable': True},
      'inbound_nodes': [[['block11_sepconv3_bn', 0, 0, {}],
        ['add_8', 0, 0, {}]]],
      'name': 'add_9'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block12_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_9', 0, 0, {}]]],
      'name': 'block12_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 357},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block12_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block12_sepconv1_act', 0, 0, {}]]],
      'name': 'block12_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block12_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block12_sepconv1', 0, 0, {}]]],
      'name': 'block12_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block12_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block12_sepconv1_bn', 0, 0, {}]]],
      'name': 'block12_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 368},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block12_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block12_sepconv2_act', 0, 0, {}]]],
      'name': 'block12_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block12_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block12_sepconv2', 0, 0, {}]]],
      'name': 'block12_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block12_sepconv3_act',
       'trainable': True},
      'inbound_nodes': [[['block12_sepconv2_bn', 0, 0, {}]]],
      'name': 'block12_sepconv3_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 379},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block12_sepconv3',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block12_sepconv3_act', 0, 0, {}]]],
      'name': 'block12_sepconv3'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block12_sepconv3_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block12_sepconv3', 0, 0, {}]]],
      'name': 'block12_sepconv3_bn'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_10', 'trainable': True},
      'inbound_nodes': [[['block12_sepconv3_bn', 0, 0, {}],
        ['add_9', 0, 0, {}]]],
      'name': 'add_10'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block13_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['add_10', 0, 0, {}]]],
      'name': 'block13_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 391},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 728,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block13_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block13_sepconv1_act', 0, 0, {}]]],
      'name': 'block13_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block13_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block13_sepconv1', 0, 0, {}]]],
      'name': 'block13_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block13_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block13_sepconv1_bn', 0, 0, {}]]],
      'name': 'block13_sepconv2_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 402},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 1024,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block13_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block13_sepconv2_act', 0, 0, {}]]],
      'name': 'block13_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block13_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block13_sepconv2', 0, 0, {}]]],
      'name': 'block13_sepconv2_bn'},
     {'class_name': 'Conv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros', 'config': {}},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 1024,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (1, 1),
       'name': 'conv2d_3',
       'padding': 'same',
       'strides': (2, 2),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['add_10', 0, 0, {}]]],
      'name': 'conv2d_3'},
     {'class_name': 'MaxPooling2D',
      'config': {'data_format': 'channels_last',
       'dtype': 'float32',
       'name': 'block13_pool',
       'padding': 'same',
       'pool_size': (3, 3),
       'strides': (2, 2),
       'trainable': True},
      'inbound_nodes': [[['block13_sepconv2_bn', 0, 0, {}]]],
      'name': 'block13_pool'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'batch_normalization_3',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['conv2d_3', 0, 0, {}]]],
      'name': 'batch_normalization_3'},
     {'class_name': 'Add',
      'config': {'dtype': 'float32', 'name': 'add_11', 'trainable': True},
      'inbound_nodes': [[['block13_pool', 0, 0, {}],
        ['batch_normalization_3', 0, 0, {}]]],
      'name': 'add_11'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 422},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 1536,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block14_sepconv1',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['add_11', 0, 0, {}]]],
      'name': 'block14_sepconv1'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block14_sepconv1_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block14_sepconv1', 0, 0, {}]]],
      'name': 'block14_sepconv1_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block14_sepconv1_act',
       'trainable': True},
      'inbound_nodes': [[['block14_sepconv1_bn', 0, 0, {}]]],
      'name': 'block14_sepconv1_act'},
     {'class_name': 'SeparableConv2D',
      'config': {'activation': 'linear',
       'activity_regularizer': None,
       'bias_constraint': None,
       'bias_initializer': {'class_name': 'Zeros',
        'config': {},
        'shared_object_id': 433},
       'bias_regularizer': None,
       'data_format': 'channels_last',
       'depth_multiplier': 1,
       'depthwise_constraint': None,
       'depthwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'depthwise_regularizer': None,
       'dilation_rate': (1, 1),
       'dtype': 'float32',
       'filters': 2048,
       'groups': 1,
       'kernel_constraint': None,
       'kernel_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'kernel_regularizer': None,
       'kernel_size': (3, 3),
       'name': 'block14_sepconv2',
       'padding': 'same',
       'pointwise_constraint': None,
       'pointwise_initializer': {'class_name': 'GlorotUniform',
        'config': {'seed': 42}},
       'pointwise_regularizer': None,
       'strides': (1, 1),
       'trainable': True,
       'use_bias': False},
      'inbound_nodes': [[['block14_sepconv1_act', 0, 0, {}]]],
      'name': 'block14_sepconv2'},
     {'class_name': 'BatchNormalization',
      'config': {'axis': 3,
       'beta_constraint': None,
       'beta_initializer': {'class_name': 'Zeros', 'config': {}},
       'beta_regularizer': None,
       'center': True,
       'dtype': 'float32',
       'epsilon': 0.001,
       'gamma_constraint': None,
       'gamma_initializer': {'class_name': 'Ones', 'config': {}},
       'gamma_regularizer': None,
       'momentum': 0.99,
       'moving_mean_initializer': {'class_name': 'Zeros', 'config': {}},
       'moving_variance_initializer': {'class_name': 'Ones', 'config': {}},
       'name': 'block14_sepconv2_bn',
       'scale': True,
       'trainable': True},
      'inbound_nodes': [[['block14_sepconv2', 0, 0, {}]]],
      'name': 'block14_sepconv2_bn'},
     {'class_name': 'Activation',
      'config': {'activation': 'relu',
       'dtype': 'float32',
       'name': 'block14_sepconv2_act',
       'trainable': True},
      'inbound_nodes': [[['block14_sepconv2_bn', 0, 0, {}]]],
      'name': 'block14_sepconv2_act'}],
    'name': 'xception',
    'output_layers': [['block14_sepconv2_act', 0, 0]]},
   'inbound_nodes': [[['cast_to_float32', 0, 0, {}]]],
   'name': 'xception'},
  {'class_name': 'GlobalAveragePooling2D',
   'config': {'data_format': 'channels_last',
    'dtype': 'float32',
    'keepdims': False,
    'name': 'global_average_pooling2d',
    'trainable': True},
   'inbound_nodes': [[['xception', 1, 0, {}]]],
   'name': 'global_average_pooling2d'},
  {'class_name': 'Dropout',
   'config': {'dtype': 'float32',
    'name': 'dropout',
    'noise_shape': None,
    'rate': 0.5,
    'seed': 42,
    'trainable': True},
   'inbound_nodes': [[['global_average_pooling2d', 0, 0, {}]]],
   'name': 'dropout'},
  {'class_name': 'Dense',
   'config': {'activation': 'linear',
    'activity_regularizer': None,
    'bias_constraint': None,
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'bias_regularizer': None,
    'dtype': 'float32',
    'kernel_constraint': None,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': 42}},
    'kernel_regularizer': None,
    'name': 'dense',
    'trainable': True,
    'units': 1,
    'use_bias': True},
   'inbound_nodes': [[['dropout', 0, 0, {}]]],
   'name': 'dense'},
  {'class_name': 'Activation',
   'config': {'activation': 'sigmoid',
    'dtype': 'float32',
    'name': 'classification_head_1',
    'trainable': True},
   'inbound_nodes': [[['dense', 0, 0, {}]]],
   'name': 'classification_head_1'}],
 'name': 'model',
 'output_layers': [['classification_head_1', 0, 0]]}
    
   
    
    return noise_input_config

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
