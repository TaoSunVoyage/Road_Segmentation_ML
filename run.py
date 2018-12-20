#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import os
import tensorflow as tf
tf.set_random_seed(1)

from keras.models import load_model

from data import testGenerator, save_result
from losses import dice_loss
from metrics import f1
from mask_to_submission import make_submission


TEST_SIZE = 50
test_path = os.path.join("data", "test_set_images")
if not os.path.exists(test_path):
    raise FileNotFoundError("Please download test images!")

predict_path = "predict_images"
submission_path = "submission"
weight_path = "weights"
weight_list = ["weights_32.h5", "weights_64.h5", "weights_dilated.h5" ]

print("Check weights...")
missing_weight = list(set(weight_list) - set(os.listdir(weight_path)))
if len(missing_weight):
    raise FileNotFoundError("Can not find: " + str(missing_weight))

print("Load models and predict...")
results = 0
for w in weight_list:
    print("...Load " + w + "...")
    model = load_model(os.path.join(weight_path, w), custom_objects={"dice_loss": dice_loss, "f1": f1})
    print("...Predict...")
    testGene = testGenerator(test_path)
    results += model.predict_generator(testGene, TEST_SIZE, verbose=1)
results /= len(weight_list)
save_result(predict_path, results)

print("Make submission...")
make_submission(predict_path, test_size=TEST_SIZE, submission_filename=os.path.join(submission_path, "submission.csv"))

print("Done!")
