#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from data import build_train_val, trainvalGenerator, testGenerator, save_result
from model import unet
from losses import dice_loss
from mask_to_submission import make_submission

# paths
train_path = "data/training"
val_path = "data/validation"
test_path = "data/test_set_images"

predict_path = "predict_images"


print("Build training and validation data set...")
build_train_val(train_path, val_path, val_size=0.2)


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
# Build model
model = unet(n_filter=32, activation='elu', dropout_rate=0.2, loss=dice_loss)
# Callback functions
callbacks = [
    EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4),
    ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, verbose=1),
    TensorBoard(log_dir='tensorboard/', write_graph=True, write_images=True)
]
# Training
history = model.fit_generator(generator=trainGen, steps_per_epoch=1000,
                              validation_data=valGen, validation_steps=80,
                              epochs=50, callbacks=callbacks)


print("Predict and save results...")
testGene = testGenerator(test_path)
result = model.predict_generator(testGene, 50, verbose=1)
save_result(predict_path, result)


print("Make submission...")
make_submission(predict_path, submission_filename="submission.csv")


print("Done!")