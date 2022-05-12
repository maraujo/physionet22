#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn import preprocessing
import tensorflow_addons as tfa
import openl3


from sklearn.metrics import classification_report

from urllib.error import HTTPError
from urllib.request import urlopen, Request
from urllib.parse import urlparse 

import hashlib
import shutil
import tempfile
from tqdm import tqdm
from helper_code import *
from pathlib import Path

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

POS = "position"
TIME = "time"
TIME_COUNTER = "time_counter"
LABEL = "label"
ID = "ID"
NAVALUE = -1
#TEMP_FILE = "/tmp/cache_challege_df.pkl"
TEMP_FILE = "./cache_challege_df_long_extra_2d.pkl"
SECONDS_PER_EMBEDDING = 2
classes = ['Absent', 'Present', 'Unknown']  # Do not change the order
UNK_OPT = "positive"
EMBEDDING_SIZE = 512
main_model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music",  embedding_size=EMBEDDING_SIZE)
TRAIN_SMALL_SAMPLE = False


def extract_embbeding(recordings, frequencies, hop_size, num_locations, recording_information, murmur_locations):

    # Extract Embeedings
    embs_part, tss_part = openl3.get_audio_embedding(recordings, frequencies, hop_size=hop_size,
                                                     batch_size=4, verbose=1, model=main_model)

    # Prepare Features DataFrame
    audios_embs = []
    for j in range(num_locations):
        entries = recording_information[j].split(' ')
        recording_pos = entries[0]
        # audio_embs_df = pd.DataFrame(embs_part[j].reshape(-1))
        # audio_embs_df = audio_embs_df.transpose()

        audio_embs_df = pd.DataFrame(embs_part[j])
        audio_embs_df[POS] = recording_pos
        audio_embs_df[TIME] = tss_part[j]
        audio_embs_df[TIME_COUNTER] = range(len(tss_part[j]))
        if murmur_locations:
            audio_embs_df["has_murmur"] = recording_pos in murmur_locations
        audios_embs.append(audio_embs_df)

    audios_embs_df = pd.concat(audios_embs)
    return audios_embs_df.reset_index(drop=True)


def get_embs_df_from_patient_data_long(num_patient_files, patient_files, data_folder, verbose):
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

        audios_embs_df = extract_embbeding(recordings, frequencies, SECONDS_PER_EMBEDDING, num_locations, recording_information, murmur_locations)

        # Extract labels and use one-hot encoding.
        current_labels = np.zeros(len(classes), dtype=int)
        label = get_murmur(current_patient_data)
        audios_embs_df["augmented"] = False
        audios_embs_df["seconds"] = SECONDS_PER_EMBEDDING

        # if label == "Present":
        for extra in [SECONDS_PER_EMBEDDING-1, SECONDS_PER_EMBEDDING-0.5, SECONDS_PER_EMBEDDING+0.5,
                      SECONDS_PER_EMBEDDING+1, SECONDS_PER_EMBEDDING+1.5, SECONDS_PER_EMBEDDING+2]:
            extradf = extract_embbeding(recordings, frequencies, extra, num_locations, recording_information, murmur_locations)
            extradf["augmented"] = True
            extradf["seconds"] = extra
            audios_embs_df = pd.concat([audios_embs_df, extradf])

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

class SimpleCNNBlock(Model):
    def __init__(self, filters, kernelsize, cnnact="relu", batchnorm=False, maxpool=False):
        super().__init__()

        self.use_batchnorm = batchnorm
        self.use_maxpool = maxpool
        self.cnn = layers.Conv1D(filters, kernelsize, activation=cnnact, padding='same')
        self.batchnorm = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling1D(2)

    def call(self, input, training=False):
        x = self.cnn(input)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        return x

class SimpleCNNBlock2D(Model):
    def __init__(self, filters, kernelsize, cnnact="relu", batchnorm=False, maxpool=False):
        super().__init__()

        self.use_batchnorm = batchnorm
        self.use_maxpool = maxpool
        self.cnn = layers.Conv2D(filters, kernelsize, activation=cnnact, padding='same')
        self.batchnorm = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D((2,2))

    def call(self, input, training=False):
        x = self.cnn(input)
        if self.use_batchnorm:
            x = self.batchnorm(x)
        if self.use_maxpool:
            x = self.maxpool(x)
        return x

class ResidualBlock1D(Model):

    def __init__(self, filters, kernelsize):
        super().__init__()

        self.cnn1 = layers.Conv1D(filters, kernelsize, activation=layers.LeakyReLU(), padding='same')
        self.cnn2 = layers.Conv1D(filters, kernelsize, activation=layers.LeakyReLU(), padding='same')
        self.cnn3 = layers.Conv1D(filters, kernelsize, padding='same')
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.averagepool = layers.AveragePooling1D()
        self.lastbatchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, input, training=False):
        x1 = self.batchnorm1(self.cnn1(input))
        x1 = self.batchnorm2(self.cnn2(x1))

        x2 = self.cnn3(input)

        out = self.add([x1, x2])
        out = self.relu(out)
        out = self.lastbatchnorm(out)
        return out

class ResidualBlock2D(Model):

    def __init__(self, filters, kernelsize):
        super().__init__()

        self.cnn1 = layers.Conv2D(filters, kernelsize, activation=layers.LeakyReLU(), padding='same')
        self.cnn2 = layers.Conv2D(filters, kernelsize, activation=layers.LeakyReLU(), padding='same')
        self.cnn3 = layers.Conv2D(filters, kernelsize, padding='same')
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.averagepool = layers.AveragePooling2D()
        self.lastbatchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, input, training=False):
        x1 = self.batchnorm1(self.cnn1(input))
        x1 = self.batchnorm2(self.cnn2(x1))

        x2 = self.cnn3(input)

        out = self.add([x1, x2])
        out = self.relu(out)
        out = self.lastbatchnorm(out)
        return out

class MyNetwork(Model):
    def __init__(self, outputsize, internal_dim=4):
        super().__init__()
        self.outputsize = outputsize

        self.masked_layer = layers.Masking(mask_value=NAVALUE)


        self.cnnblock1 = SimpleCNNBlock2D(internal_dim, (3, 3), "relu", batchnorm=False, maxpool=False)
        self.cnnblock2 = SimpleCNNBlock2D(internal_dim, (5, 5), "softplus", batchnorm=False, maxpool=False)
        self.cnnblock3 = SimpleCNNBlock2D(internal_dim, (7, 7), "relu", batchnorm=False, maxpool=False)
        self.cnnblock4 = SimpleCNNBlock2D(internal_dim, (7, 7), layers.ThresholdedReLU(theta=1.0), batchnorm=False, maxpool=False)

        # self.cnn2 = layers.Conv1D(internal_dim, 5, activation='softplus', name="CNN2", padding='same')
        # self.cnn3 = layers.Conv1D(internal_dim, 7, activation='relu', name="CNN3", padding='same')
        # self.cnn4 = layers.Conv1D(internal_dim, 21, activation='softplus', name="CNN4", padding='same')
        # self.cnn5 = layers.Conv1D(internal_dim, 9, activation='relu', name="CNN5", padding='same')
        # self.cnn6 = layers.Conv1D(internal_dim, 4, activation='softplus', name="CNN6", padding='same')
        # self.cnn7 = layers.Conv1D(internal_dim, 7, activation='tanh', name="CNN7", padding='same')
        # self.cnn8 = layers.Conv1D(internal_dim, 11, activation='relu', name="CNN8", padding='same')
        # self.cnn9 = layers.Conv1D(internal_dim, kernel_size=5, padding='same', activation=layers.ThresholdedReLU(theta=1.0))

        self.resblock1 = ResidualBlock2D(internal_dim, (5, 5))
        self.resblock2 = ResidualBlock2D(internal_dim, (7, 7))
        self.resblock3 = ResidualBlock2D(internal_dim, (11, 11))

        self.maxpool = layers.MaxPooling2D((2, 2))
        self.gmaxpool = layers.GlobalMaxPooling2D()

        self.batchnorm = layers.BatchNormalization()

        #
        self.flatten = layers.Flatten()
        self.linear1 = layers.Dense(32, activation='relu', name="Linear1")
        #self.linear2 = layers.Dense(128, activation='relu', name="Linear2")
        #self.linear3 = layers.Dense(32, activation='relu', name="Linear3")
        #
        self.dropout30 = layers.Dropout(0.30)
        self.dropout20 = layers.Dropout(0.20)
        self.dropout50 = layers.Dropout(0.50)
        self.dropout80 = layers.Dropout(0.80)
        # self.lstm = layers.Bidirectional(layers.LSTM(internal_dim, return_sequences=True, dropout=.2))
        # self.lstm = layers.LSTM(internal_dim)
        #
        self.outputlayer = layers.Dense(self.outputsize, activation='softmax', name="DenseSoftMax")
        # self.cnnoutlayer = tf.keras.layers.Conv1D(3, 1, activation='softmax', padding='same', name="CNNout")


    # def summary(self):
    #     x = tf.keras.layers.Input(shape=(None, 16 * EMBEDDING_SIZE, 4))
    #     model = Model(inputs=[x], outputs=self.call(x))
    #     return model.summary()

    def call(self, input, training=False):

        masked = self.masked_layer(input)

        x1 = self.cnnblock1(masked)
        x1 = self.dropout80(x1)
        x2 = self.cnnblock2(masked)
        x2 = self.dropout80(x2)
        x3 = self.cnnblock3(masked)
        x3 = self.dropout80(x3)
        x4 = self.cnnblock4(masked)
        x4 = self.dropout80(x4)

        # x1 = self.batchnorm(self.cnn1(masked))
        # x2 = self.batchnorm(self.cnn2(masked))
        # x3 = self.batchnorm(self.cnn3(masked))
        # x4 = self.batchnorm(self.cnn4(masked))
        # x5 = self.batchnorm(self.cnn5(masked))
        # x6 = self.batchnorm(self.cnn6(masked))
        # x7 = self.batchnorm(self.cnn7(masked))
        # x8 = self.batchnorm(self.cnn8(masked))
        # x9 = self.batchnorm(self.cnn9(masked))

        resout = self.resblock1(x1 + x2 + x3 + x4)
        resout = self.dropout50(resout)
        resout = self.resblock2(resout)
        resout = self.dropout50(resout)
        resout = self.resblock3(resout)
        resout = self.dropout50(resout)
        # print("Shape resout:", resout.shape)
        #
        # # concat = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6, x7, x8, x9, resout])
        # concat = tf.keras.layers.Concatenate(axis=2)([x1, x2, x3, x4, resout])
        # resout = self.lstm(resout)

        #x = self.lstm(concat)
        # print("Shape 2:", concat.shape)
        x = self.gmaxpool(resout)
        # x = self.flatten(masked)

        # print("Shape 3:", x.shape)
        # x = self.linear1(x)
        out = self.outputlayer(x)

        return out


def do_network_training(X_train, X_test, y_train, y_test):

    MAX_EPOCHS = 300
    #MAX_EPOCHS = 10

    model = MyNetwork(y_train.shape[1])

    lr = 0.0001
    patience = 12

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=100000, decay_rate=0.98, staircase=True)

    earlystopper = tf.keras.callbacks.EarlyStopping(patience=patience, monitor='val_loss', mode='auto',
                                                    min_delta=0, verbose=0)

    model.compile(optimizer=tfa.optimizers.AdamW(lr_schedule),
                    #loss=tfa.losses.SigmoidFocalCrossEntropy(),
                    loss=losses.CategoricalCrossentropy(label_smoothing=0.1)
                    )

    # class_weight = {0: 1.,
    #                 1: 1.,
    #                 2: 25.}
    class_weight = {0: 1.3,
                    1: 1,
                    }

    with tf.device('/cpu:0'):
        model.fit(X_train, y_train,
                  epochs=MAX_EPOCHS,
                  shuffle=True,
                  batch_size=64,
                  validation_data=(X_test, y_test),
                  callbacks=[earlystopper],
                  class_weight=class_weight,
                  )

    model.summary()


    return model

def regroup(df, fillna_value=NAVALUE):

    #new_idx = [(id, time, pos) for id in df[ID].unique() for pos in ['AV', 'PV', 'TV', 'MV'] for time in df[TIME_COUNTER].unique()]
    new_idx = [(id, time, pos) for id in df[ID].unique() for pos in ['AV', 'PV', 'TV', 'MV'] for time in range(32)]
    df = df.drop_duplicates(subset=[ID, TIME_COUNTER, POS])
    df = df.set_index([ID, TIME_COUNTER, POS]).reindex(new_idx).reset_index()
    df[classes] = df.groupby(ID)[classes].fillna(method="ffill").fillna(method="bfill")
    return df.fillna(fillna_value)

def get_numpy(df_in, get_y=True, fillna_value=NAVALUE):
    if UNK_OPT in ["positive", "remove"]:
        considered_classes = ['Absent', 'Present']
    else:
        considered_classes = classes

    # In case this dataframe does not have all cols, we add them on the fly here.
    # if 16 * EMBEDDING_SIZE not in df:
    #     df = df.reindex(columns=["ID"] + list(range(16 * EMBEDDING_SIZE))).fillna(fillna_value)
    df = df_in.copy()
    # for r in range(EMBEDDING_SIZE):
    #     df[r] = df[r].mean() / df[r].std()

    xt = df.loc[:, range(EMBEDDING_SIZE)].values
    #xt = (xt.mean() / xt.std())

    if get_y:
        yt = df.loc[:, considered_classes].astype(int).values

    g = df.reset_index().groupby([ID, POS])
    xtg = [xt[i.values, :] for k, i in g.groups.items()]
    if get_y:
        ytg = [yt[i.values, :] for k, i in g.groups.items()]
        ytg = np.array(ytg)
        ytg = ytg.reshape(df[ID].unique().shape[0], -1)[:, 0:len(considered_classes)]

    xtg = np.array(xtg)
    # xtg = xtg.reshape(-1, 4, df[TIME_COUNTER].unique().shape[0], 512)
    xtg = xtg.reshape(-1, 4, 32, 512)
    # xtg = np.swapaxes(xtg, 1, 2)

    if get_y:
        return xtg, ytg
    else:
        return xtg

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose, submission=False):
    # What should we do with Unknown (?): options "nothing", "positive", "remove"

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

    if os.path.exists(TEMP_FILE) and False:
        print(" ----------------------------   Loading TEMP file! ------------------------------------------------")
        print(TEMP_FILE)
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
    else:
        all_patients_embs_df = get_embs_df_from_patient_data_long(num_patient_files, patient_files, data_folder, verbose)
        try:
            all_patients_embs_df.to_pickle(TEMP_FILE)
        except:
            print("Could not save TEMP file")

    #Stratify Sample
    all_patients_embs_df.sort_values(by=["ID", "position"]).reset_index(drop=True)

    if UNK_OPT == "positive":
        # all_patients_embs_df.loc[all_patients_embs_df["Unknown"] == True, "has_murmur"] = True   # We could assume that unknown is positive label
        all_patients_embs_df.loc[all_patients_embs_df["Unknown"] == True, "Present"] = True  # We could assume that unknown is positive label
        all_patients_embs_df.loc[all_patients_embs_df["Unknown"] == True, "Unknown"] = False
    elif UNK_OPT == "remove":
        all_patients_embs_df = all_patients_embs_df[all_patients_embs_df["Unknown"] != True]  # Or we could just drop these instances
    else:
        pass

    seed = 42
    np.random.seed(seed)

    all_patients_embs_df = all_patients_embs_df[(all_patients_embs_df["augmented"] == False) |
                                                ((all_patients_embs_df["Present"] == True) & (all_patients_embs_df["augmented"] == True))
                                                ]  # Or we could just drop these instances
    pids = all_patients_embs_df["ID"].unique()
    np.random.shuffle(pids)

    train_ids = set(pids[:int(pids.shape[0]*.8)])
    test_ids = set(pids) - train_ids

    train_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(train_ids)]
    test_data = all_patients_embs_df[all_patients_embs_df["ID"].isin(test_ids)]

    train_data = train_data.reset_index(drop=True)
    train_data.loc[train_data["seconds"] != 2, "ID"] = train_data[train_data["seconds"] != 2][["ID", "seconds"]].apply(lambda x: str(x[0]) + "_" + str(x[1]), axis=1)

    train_data = regroup(train_data)  # Shape should be something like (-1, 4, 16*512)
    test_data = regroup(test_data)

    X_train, y_train = get_numpy(train_data)
    X_test, y_test = get_numpy(test_data)

    if not submission:
        copy_test_data(test_ids, "all_training_data/", "test_data/")

    X_train = tf.convert_to_tensor(X_train, tf.float32)
    X_test = tf.convert_to_tensor(X_test, tf.float32)
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    model = do_network_training(X_train, X_test, y_train, y_test)

    preds = model.predict(X_test).argmax(axis=1)
    labels = y_test.argmax(axis=1)

    print("Predictions:", preds)
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))

    # Save the model.
    save_challenge_model(model_folder, classes, model)

    if verbose >= 1:
        print('Done.')

def run_challenge_model(model, data, recordings, verbose, thresholds=None, use_cache=True):

    fields = data.split("\n")[0].split(" ")

    pid = fields[0]
    num_locations = int(fields[1])
    frequency_sample_rate = int(fields[2])
    recording_information = data.split("\n")[1: 1+num_locations]

    # Debugging
    # if pid != "85108":
    #     return classes,  [1,0,0],  [1,0,0]

    if use_cache and False:
        all_patients_embs_df = pd.read_pickle(TEMP_FILE)
        df_input = all_patients_embs_df[all_patients_embs_df["ID"] == str(pid)]
        df_input = df_input[df_input["augmented"] != True]
    else:

        df_input = extract_embbeding(recordings, [frequency_sample_rate] * len(recordings), SECONDS_PER_EMBEDDING, num_locations, recording_information, murmur_locations=None)
        df_input["ID"] = pid
        for col in ['Unknown', 'Present', 'Absent']:
            df_input[col] = 0

    test_data = regroup(df_input)
    X_test = get_numpy(test_data, get_y=False)
    X_test = tf.convert_to_tensor(X_test, tf.float32)
    
    
    probs = model.predict(X_test)[0]

    pred = np.zeros(3)
    if probs[1] >= 0.49:
        pred[1] = 1
    elif probs[1] > 0.45:
        pred[2] = 1
    else:
        if df_input["Present"].sum() > 1:
            print("PID:", pid)
        pred[0] = 1

    if UNK_OPT in ["positive", "remove"]:
        # probs would have only the labels to "absent", "present". Here we append the prob of 0.0 to unknown
        # An alternative is assingin the unknown class in case "present" or "absent" is no so strong (i.e., both close to 0.5)
        if pred[2] != 1:
            probs = np.append(probs, [0])
        else:
            probs = np.append(probs - 0.2, [0.4])

    return classes, pred, list(probs)

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    from keras.models import load_model
    return load_model(os.path.join(model_folder, 'keras_model'))
    # load_config(os.path.join(model_folder, 'config.sav'))
    # return load_model(os.path.join(model_folder, 'model.sav'))


# Save your trained model.
def save_challenge_model(model_folder, classes, classifier):
    classifier.save(os.path.join(model_folder, 'keras_model'))

    # filename = os.path.join(model_folder, 'model.sav')
    # save_model(classifier, filename)
    # save_config(os.path.join(model_folder, 'config.sav'))


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