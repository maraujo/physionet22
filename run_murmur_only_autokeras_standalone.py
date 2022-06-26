# -*- coding: utf-8 -*-
from keras.models import load_model
from autokeras.keras_layers import CastToFloat32
from loguru import logger
from tqdm.notebook import tqdm
import os
import glob
import pandas as pd
import shutil
import tensorflow as tf
import numpy as np
import autokeras as ak
import random

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

aws_id = input("AWS ID: ")
secret_id = input("Secret ID: ")
os.environ['AWS_ACCESS_KEY_ID'] = aws_id
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_id

os.system("aws s3 cp s3://1hh-algorithm-dev/preprocessing/train_ready_murmur.tar.gz ./train_folder.tar.gz")
os.system("tar -xf train_folder.tar.gz")
os.system("aws s3 cp s3://1hh-algorithm-dev/preprocessing/val_ready_murmur.tar.gz ./val_folder.tar.gz")
os.system("tar -xf val_folder.tar.gz")
os.system("aws s3 cp s3://1hh-algorithm-dev/preprocessing/test_ready_murmur.tar.gz ./test_folder.tar.gz")
os.system("tar -xf test_folder.tar.gz")

data_folder = "output"
model_folder = "model"
verbose = True
train_folder = "train_folder"
test_folder = "test_folder"
val_folder = "val_folder"
val_positive_folder = val_folder + os.path.sep + "positive"
val_negative_folder = val_folder + os.path.sep + "negative"
test_positive_folder = test_folder + os.path.sep + "positive"
test_negative_folder = test_folder + os.path.sep + "negative"
train_positive_folder = train_folder + os.path.sep + "positive"
train_negative_folder = train_folder + os.path.sep + "negative"

train_outcome_folder = "train_outcome_folder"
test_outcome_folder = "test_outcome_folder"
val_outcome_folder = "val_outcome_folder"
val_outcome_positive_folder = val_outcome_folder + os.path.sep + "positive"
val_outcome_negative_folder = val_outcome_folder + os.path.sep + "negative"
test_outcome_positive_folder = test_outcome_folder + os.path.sep + "positive"
test_outcome_negative_folder = test_outcome_folder + os.path.sep + "negative"
train_outcome_positive_folder = train_outcome_folder + os.path.sep + "positive"
train_outcome_negative_folder = train_outcome_folder + os.path.sep + "negative"

WORKERS = min(os.cpu_count() - 1, 8)

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

class MatheusGraph(graph.Graph):
  def _compile_keras_model(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer_name = hp.Choice(
            "optimizer",
            ["adam"],
            default="adam",
        )
        # TODO: add adadelta optimizer when it can optimize embedding layer on GPU.
        learning_rate = hp.Choice(
            "learning_rate", [1e-3], default=1e-3
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

class MatheusAutoModel(ak.AutoModel):

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
        return MatheusGraph(inputs=inputs, outputs=outputs)

  def _build_graph(self):
        # Using functional API.
        if all([isinstance(output, node_module.Node) for output in self.outputs]):
            graph = MatheusGraph(inputs=self.inputs, outputs=self.outputs)
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

import keras_tuner as kt
rescale_factor = 3
img_height = int(216 / rescale_factor)
img_width = int(216 / rescale_factor)
batch_size = 32
class_weight = {}
MODEL_DESTINY = "./model_destiny/"   

train_data = ak.image_dataset_from_directory(
    train_folder,
    # Use 20% data as testing data.
    # validation_split=0.5,
    subset="training",
    # Set seed to ensure the same split when loading testing data.
    seed=42,
    image_size=(img_height, img_width)
)

test_data = ak.image_dataset_from_directory(
    test_folder,
    #validation_split=0.2,
    subset="test",
    seed=42,
    image_size=(img_height, img_width)
)

val_data = ak.image_dataset_from_directory(
    val_folder,
    #validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width)
)


from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(
  log_dir= MODEL_DESTINY + 'logs',
  histogram_freq=1,
  write_images=True,
  embeddings_freq = 10
)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard
# %tensorboard --logdir "/content/drive/MyDrive/One Heart Health/murmur_detection/logs" --reload_multifile=true

# Deep learning model

def run_model(train_dataset, val_dataset, test_dataset, task):
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            

  AUC_metric = tf.keras.metrics.AUC(name="auc", curve="PR")
  AUC_ROC_metric = tf.keras.metrics.AUC(name="auc_roc", curve="ROC")
  FN_metric = tf.keras.metrics.FalseNegatives(name="fn")
  FP_metric = tf.keras.metrics.FalsePositives(name="fp")
  ACC_metric = tf.keras.metrics.Accuracy()
  PRECISION_metric = tf.keras.metrics.Precision()
  RECALL_metric = tf.keras.metrics.Recall()
  all_metrics = [AUC_metric, FN_metric, FP_metric, ACC_metric, PRECISION_metric, RECALL_metric, AUC_ROC_metric]
  class_weight = {0: 1, 1: 1.5}

  # input_node = ak.ImageInput()
  # output_node = ak.ConvBlock(
  #     kernel_size=None,
  #     num_blocks=None,
  #     num_layers=None,
  #     filters=None,
  #     max_pooling=None,
  #     separable=None,
  #     dropout=None
  # )(input_node)
  # output_node = ak.ClassificationHead()(output_node)
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=MODEL_DESTINY + "{}_model_checkpoint_{}_x_{}_".format(img_height, img_width,task) + "{epoch:02d}-{val_auc:.2f}-{val_auc_roc:.2f}-{val_fn:.2f}-{val_fp:.2f}.model",
      save_weights_only=False,
      monitor='val_auc',
      loss="binary_crossentropy",
      mode='max',
      initial_value_threshold=0.75,
      save_best_only=True)

  input_node = ak.ImageInput()
  hp = kt.HyperParameters()
  output_node = ak.ImageBlock(
    block_type="xception",
    augment=False
)(input_node)
  # output_node = ak.XceptionBlock()(input_node)
  output_node = ak.SpatialReduction(reduction_type="flatten")(output_node)
  output_node = ak.DenseBlock(num_layers=1, num_units=64, dropout=0)(output_node)
  output_node = ak.ClassificationHead(dropout=0)(output_node)

  early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                min_delta=0.01,
                patience=15,
                verbose=1,
                mode="max",
                baseline=0.55,
                restore_best_weights=True,
            )
  
  clf = MatheusAutoModel(
      inputs=input_node, seed=42, objective=kt.Objective("val_auc", direction="max"), outputs=output_node, overwrite=False, 
      max_trials=100, tuner="bayesian", metrics = all_metrics)

  kt.engine.tuner.MAX_FAIL_STREAK = 1000
  clf.fit(train_dataset, workers=WORKERS, callbacks=[model_checkpoint_callback, early_stopping, tensorboard], max_queue_size=50000, epochs=100,  batch_size=128, validation_data=val_dataset, class_weight=class_weight)
  print(clf.evaluate(test_dataset))

  print(clf.evaluate(test_data, return_dict=True))
  model2 = clf.export_model()
  try:
      model2.save(MODEL_DESTINY + "model_autokeras_murmur_detection_final.model", save_format="tf")
  except Exception:
      model2.save(MODEL_DESTINY + "model_autokeras_murmur_detection_final.h5")
  print("Done")
run_model(train_data, val_data, test_data, "murmur")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





