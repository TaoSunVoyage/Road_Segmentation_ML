#!/usr/bin/env python3

# credits: https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/losses.py

from keras.losses import binary_crossentropy
from metrics import dice_coeff

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss