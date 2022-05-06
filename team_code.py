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
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from resnet import ResNet1d
from evaluate_model import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data.dataset import Dataset
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
SEQ_LENGTH = 24576
# DEVICE = 'cpu'
DEVICE = 'cuda:0'
EPOCHS = 160

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
        self.identity = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.lin = nn.Linear(last_layer_dim, n_classes)
        self.lin = nn.Linear(n_filters_last, n_classes)
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

        x = self.identity(x)
        x = self.avgpool(x)
        # Flatten array
        x = torch.flatten(x, 1)
        # Fully conected layer
        x = self.lin(x)
        return x

# Split training data into training and testing
def split_training_data(data_folder, save_folder):
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files==0:
        raise Exception('No data was provided.')

    X_train, X_test, y_train, y_test = train_test_split(patient_files, range(len(patient_files)), test_size=0.2, random_state=42)

    if os.path.exists(os.path.join(save_folder, 'train')):
        shutil.rmtree(os.path.join(save_folder, 'train'))
    os.makedirs(os.path.join(save_folder, 'train'),exist_ok=True)
    for i in range(len(X_train)):
        current_patient_data = load_patient_data(X_train[i])
        patient_id = get_patient_id(current_patient_data)
        shutil.copy(os.path.join(data_folder, patient_id+'.txt'), os.path.join(save_folder, 'train'))
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            for i in range(1, len(entries)):
                shutil.copy(os.path.join(data_folder, entries[i]), os.path.join(save_folder, 'train'))

    if os.path.exists(os.path.join(save_folder, 'test')):
        shutil.rmtree(os.path.join(save_folder, 'test'))
    os.makedirs(os.path.join(save_folder, 'test'),exist_ok=True)
    for i in range(len(X_test)):
        current_patient_data = load_patient_data(X_test[i])
        patient_id = get_patient_id(current_patient_data)
        shutil.copy(os.path.join(data_folder, patient_id+'.txt'), os.path.join(save_folder, 'test'))
        num_locations = get_num_locations(current_patient_data)
        recording_information = current_patient_data.split('\n')[1:num_locations+1]
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            for i in range(1, len(entries)):
                shutil.copy(os.path.join(data_folder, entries[i]), os.path.join(save_folder, 'test'))
    return os.path.join(save_folder, 'train'), os.path.join(save_folder, 'test')

def train_epoch(model, train_dataloader, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    epoch_iterator = tqdm(
        train_dataloader, desc="Training epoch X, iter (X / X) (loss=X.X)", dynamic_ncols=True
    )
    all_label = []
    all_output_softmax = []
    all_output_argmax = []
    for batch_idx, (data, label) in enumerate(epoch_iterator):
        data = data.float().to(DEVICE)
        label = label.long().to(DEVICE)
        # compute output
        output = model(data)
        loss = criterion(output, label) + 3 * my_loss(output, label)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record loss
        losses.update(loss.item(), data.shape[0])
        output_softmax = F.softmax(output, dim=1)
        output_argmax = torch.argmax(output_softmax, dim=1)
        all_label.append(label.detach().cpu().numpy())
        all_output_softmax.append(output_softmax.detach().cpu().numpy())
        all_output_argmax.append(output_argmax.detach().cpu().numpy())
        epoch_iterator.set_description(
            "Training Epoch %d, iter (%d/%d) (loss=%2.5f)" % (epoch, batch_idx+1, len(epoch_iterator), loss.item())
        )
    all_label = to_one_hot_bool(np.concatenate(all_label), 3)
    all_output_softmax = np.concatenate(all_output_softmax)
    all_output_argmax = to_one_hot_bool(np.concatenate(all_output_argmax), 3)
    # compute the accuracy
    classes = ['Present', 'Unknown', 'Absent']
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(all_label, all_output_softmax)
    accuracy = compute_accuracy(all_label, all_output_argmax)
    f_measure, f_measure_classes = compute_f_measure(all_label, all_output_argmax)
    challenge_score = compute_challenge_score(all_label, all_output_argmax, classes)
    accs = {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f_measure': f_measure,
        'challenge_score': challenge_score,
    }
    return losses.avg, accs

def test(model, test_folder):
    model.eval()
    patient_files = find_patient_files(test_folder)
    num_patient_files = len(patient_files)
    classes = ['Present', 'Unknown', 'Absent']
    all_label = []
    all_output_softmax = []
    all_output_argmax = []
    # load the current patient data and recordings.
    threshold = 0.1
    with torch.no_grad():
        for i in range(num_patient_files):
            # print(f'processing patient:{patient_files[i]}')
            current_patient_data = load_patient_data(patient_files[i])
            recordings = load_recordings_normalized(test_folder, current_patient_data)
            # do prediction for each recording
            out_softmaxs = []
            out_argmaxs = []
            for recording in recordings:
                # now, let's just random pick one segment from the recording to make prediction
                current_recording_length = recording.shape[0]
                if current_recording_length > SEQ_LENGTH:
                    start_point = 0
                    steps = []
                    while (start_point + SEQ_LENGTH) < current_recording_length:
                        steps += [start_point]
                        start_point += SEQ_LENGTH # non-overlap
                    steps += [current_recording_length - SEQ_LENGTH]
                    # start_point = np.random.randint(0, current_recording_length - SEQ_LENGTH)
                    # print(f'steps:{steps}')
                    recording_segments = []
                    for step in steps:
                        recording_segments.append(recording[step:step+SEQ_LENGTH])
                    recording_segment = torch.from_numpy(np.stack(recording_segments)[:,np.newaxis,:]).float().to(DEVICE)
                else: # padding with zeros
                    recording_segment = np.zeros(SEQ_LENGTH)
                    recording_segment[0:current_recording_length] = recording
                    recording_segment = torch.from_numpy(recording_segment[None,None]).float().to(DEVICE)
                # make prediction
                out = model(recording_segment)
                out_softmax = F.softmax(out, dim=1)
                out_argmax = torch.argmax(out_softmax, dim=1)
                # if there are multiple segments in this recording, the final prediction is the one with the largest 0 probability
                out_softmax_0 = out_softmax[:,0]
                max_0_index = torch.argmax(out_softmax_0, dim=0)
                out_softmaxs.append(out_softmax[max_0_index].cpu().numpy())
                out_argmaxs.append(out_argmax[max_0_index].cpu().numpy())
            # if one location predicts "has murmur", then the final prediction will be "has murmur", then for "Unknown", then for "Absent"
            all_output_argmax.append(min(out_argmaxs))
            all_output_softmax.append(out_softmaxs[out_argmaxs.index(min(out_argmaxs))])
            label = get_label(current_patient_data)
            label = classes.index(label)
            all_label.append(label)
    all_label = to_one_hot_bool(np.stack(all_label), 3)
    all_output_softmax = np.stack(all_output_softmax)
    all_output_argmax = to_one_hot_bool(np.stack(all_output_argmax), 3)
    # compute the accuracy
    classes = ['Present', 'Unknown', 'Absent']
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(all_label, all_output_softmax)
    accuracy = compute_accuracy(all_label, all_output_argmax)
    f_measure, f_measure_classes = compute_f_measure(all_label, all_output_argmax)
    challenge_score = compute_challenge_score(all_label, all_output_argmax, classes)
    accs = {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f_measure': f_measure,
        'challenge_score': challenge_score,
    }
    return accs

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    
    # download the pretrained model
    # download_model_from_url('https://www3.nd.edu/~scl/models/model.pth',model_folder,'model.pth')
    # split training data
    print('Splitting data...')
    train_data_dir, test_data_dir = split_training_data(data_folder, model_folder)
    print('Splitting data finished...')
    # define model
    model = ResNet1d(
        input_dim=(1,SEQ_LENGTH), 
        blocks_dim=list(zip([64, 128, 196, 256, 320], [8192, 2048, 512, 128, 64])), 
        n_classes=3, 
        kernel_size=17, 
        dropout_rate=0.8,
    )
    model.to(DEVICE)
    # define criterion and optimizer
    class_weights = torch.Tensor([5.0,1.0,1.0]).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    # define dataloader
    train_dataset = physionet22_dataset(data_dir=train_data_dir, recording_length=SEQ_LENGTH)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=8,
                                                    drop_last=False)
    best_challenge_score = 100000
    print('Start training...')
    for epoch in range(EPOCHS):
        training_loss, training_accs = train_epoch(model, train_dataloader, criterion, optimizer, epoch)
        scheduler.step()
        test_accs = test(model, test_data_dir)
        for k, v in training_accs.items():
            print(f"Training: {k}:{v:.4f}")
        for k, v in test_accs.items():
            print(f"Test: {k}:{v:.4f}")
        # save the current model
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_folder, "latest.pth"))
        # save the best model so far
        if best_challenge_score > test_accs['challenge_score']:
            best_challenge_score = test_accs['challenge_score']
            torch.save(model.state_dict(), os.path.join(model_folder, "best.pth"))
        print(f'Best challenge score so far:{best_challenge_score:.4f}')
    return 0

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    # define model
    model = ResNet1d(
        input_dim=(1,SEQ_LENGTH), 
        blocks_dim=list(zip([64, 128, 196, 256, 320], [8192, 2048, 512, 128, 64])), 
        n_classes=3, 
        kernel_size=17, 
        dropout_rate=0.8,
    )
    # load the pretrained model
    model_dict = torch.load(os.path.join(model_folder, 'best.pth'), map_location='cpu')
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

# Load recordings and return dict
def load_recordings_return_dict(data_folder, data):
    # Now get the recordings in txt file
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]
    recordings_dict = {}
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        current_location = entries[0]
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recording = recording.astype(np.float)
        # do we need to do normalization?
        recording_mean = recording.mean()
        recording_std = recording.std()
        recording = (recording - recording_mean) / recording_std
        # update the recordings_dict with the loaded recording
        recordings_dict[current_location] = recording
    return recordings_dict

# Load recordings.
def load_recordings_normalized(data_folder, data):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]
    recordings = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        # do we need to do normalization?
        recording_mean = recording.mean()
        recording_std = recording.std()
        recording = (recording - recording_mean) / recording_std
        recordings.append(recording)
    return recordings

class physionet22_dataset(Dataset):

    def __init__(self, data_dir, recording_length):
        """
        data_dir: training/validation/test data folder
        recording_length: the length of recording segment we want to cut for the original recording
        """
        patient_files = find_patient_files(data_dir)
        num_patient_files = len(patient_files)
        self.recordings = []
        self.labels = []
        self.recording_length = recording_length
        classes = ['Present', 'Unknown', 'Absent']
        # load the current patient data and recordings.
        for i in range(num_patient_files):
            current_patient_data = load_patient_data(patient_files[i])
            recordings_dict = load_recordings_return_dict(data_dir, current_patient_data)
            num_locations = len(recordings_dict)
            label = get_label(current_patient_data)
            label = classes.index(label)
            # if murmur is present, only keep the locations with murmur, otherwise all locations are Unknown/Absent
            if label == 0:
                # get all the locations with murmurs
                patient_data_series = pd.Series(current_patient_data.split('\n')[num_locations+1:])
                murmur_locations = patient_data_series[patient_data_series.str.contains("Murmur location")].iloc[0].split(": ")[-1]
                murmur_locations = murmur_locations.split("+")
                for location in murmur_locations:
                    self.recordings.append(recordings_dict[location])
                    self.labels.append(label)
            # if murmur is Unkown, we do not know which location is unknown, let's just throw this patient away
            elif label == 1:
                pass
            else:
                for k, v in recordings_dict.items():
                    self.recordings.append(recordings_dict[k])
                    self.labels.append(label)

    def __getitem__(self, index):
        # get the recordings of current patient
        recording = self.recordings[index]
        current_recording_length = recording.shape[0]
        # random sample a segment from current recording, kind of like data augmentation?
        if current_recording_length > self.recording_length:
            start_point = np.random.randint(0, current_recording_length - self.recording_length)
            data = recording[start_point:start_point+self.recording_length]
        else: # do padding
            data = np.zeros(self.recording_length)
            data[0:current_recording_length] = recording
        return data[None], self.labels[index]

    def  __len__(self):
        return len(self.recordings)

# define a custumized loss function to penalized the false negatives when murmurs is presenet or unknown
def my_loss(output, target):
    output = F.log_softmax(output, dim=1)
    # convert the target into one-hot encoding
    target_onehot = F.one_hot(target, num_classes=3)
    # activate when ground truth is 0 or 1
    target_onehot_comp = 1 - target_onehot
    loss = output * target_onehot_comp
    # minimize the probability of predicting 2
    return loss[:, 2].mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count