#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import os
import tensorflow as tf
tf.set_random_seed(1)

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from data import build_train_val, trainvalGenerator, testGenerator, save_result
from model import unet, unet_dilated
from losses import dice_loss
from mask_to_submission import make_submission

NUM_EPOCH = 100
NUM_TRAINING_STEP = 1000
NUM_VALIDATION_STEP = 80
TEST_SIZE = 50

# paths
train_path = os.path.join("data", "training")
val_path = os.path.join("data", "validation")
test_path = os.path.join("data", "test_set_images")

predict_path = "predict_images"
submission_path = "submission"
weight_path = "weights"


if not os.path.exists(val_path):
    print("Build training and validation data set...")
    build_train_val(train_path, val_path, val_size=0.2)
else:
    print("Have found training and validation data set...")


print("Create generator for training and validation...")
# Arguments for data augmentation
data_gen_args = dict(rotation_range=45,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

# Build generator for training and validation set
trainGen, valGen = trainvalGenerator(batch_size=2, aug_dict=data_gen_args, 
                                     train_path=train_path, val_path=val_path,
                                     image_folder='images', mask_folder='groundtruth',
                                     train_dir = None, # Set it to None if you don't want to save
                                     val_dir = None, # Set it to None if you don't want to save
                                     target_size = (400, 400), seed = 1)


print("Build model and training...")

print("...Build & train the modified U-Net with 32 filters...")
# Build model
model_32 = unet(n_filter=32, activation='elu', dropout_rate=0.2, loss=dice_loss)
# Callback functions
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_32.h5'), monitor='val_loss', save_best_only=True, verbose=1)
]
# Training
history_32 = model_32.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks)


print("...Build & train the modified U-Net with 64 filters...")
# Build model
model_64 = unet(n_filter=64, activation='elu', dropout_rate=0.2, loss=dice_loss)
# Callback functions
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_64.h5'), monitor='val_loss', save_best_only=True, verbose=1)
]
# Training
history_64 = model_64.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                    validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                    epochs=NUM_EPOCH, callbacks=callbacks)


print("...Build & train the U-Net with dilated convolution...")
# Build model
model_dilated = unet_dilated(n_filter=32, activation='elu', loss=dice_loss, dropout=False, batchnorm=False)
# Callback functions
callbacks = [
    # EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint(os.path.join(weight_path, 'weights_dilated.h5'), monitor='val_loss', save_best_only=True, verbose=1)
]
# Training
history_dilated = model_dilated.fit_generator(generator=trainGen, steps_per_epoch=NUM_TRAINING_STEP,
                                              validation_data=valGen, validation_steps=NUM_VALIDATION_STEP,
                                              epochs=NUM_EPOCH, callbacks=callbacks)


print("Predict and save results...")
print("...For U-Net with 32 filters...")
testGene = testGenerator(test_path)
result_1 = model_32.predict_generator(testGene, TEST_SIZE, verbose=1)
print("...For U-Net with 64 filters...")
testGene = testGenerator(test_path)
result_2 = model_64.predict_generator(testGene, TEST_SIZE, verbose=1)
print("...For U-Net with dilated convolution...")
testGene = testGenerator(test_path)
result_3 = model_dilated.predict_generator(testGene, TEST_SIZE, verbose=1)
print("...Averaging the prediction results...")
result = (result_1 + result_2 + result_3)/3
save_result(predict_path, result)


print("Make submission...")
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))

print("Done!")