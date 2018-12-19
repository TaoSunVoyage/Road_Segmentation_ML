#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)

import os
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def make_dir(path):
    """Make directory."""
    if not os.path.exists(path):
        os.makedirs(path)

def build_train_val(train_path, val_path, val_size=0.2, seed=1):
    """Build training and validation set images.

    Arguments:
        train_path {str} -- path of training set
        val_path {str} -- path of validation set

    Keyword Arguments:
        val_size {float} -- size of validation set (default: {0.2})
        seed {int} -- random seed (default: {1})

    """
    # Rotate and Flip -> 8-fold dataset
    for i in range(1, 101):
        im = Image.open(os.path.join(train_path, 'images', 'satImage_%.3d.png'%i))
        ma = Image.open(os.path.join(train_path, 'groundtruth', 'satImage_%.3d.png'%i))

        im_f = im.transpose(Image.FLIP_LEFT_RIGHT)
        io.imsave(os.path.join(train_path, 'images', 'satImage_%.3d_f.png'%i), np.array(im_f))

        ma_f = ma.transpose(Image.FLIP_LEFT_RIGHT)
        io.imsave(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_f.png'%i), np.array(ma_f))
    
    
        for angle in [90, 180, 270]:
            im_r = im.rotate(angle)
            io.imsave(os.path.join(train_path, 'images', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(im_r))

            im_f_r = im_f.rotate(angle)
            io.imsave(os.path.join(train_path, 'images', 'satImage_%.3d_f_%.3d.png'%(i, angle)), np.array(im_f_r))

            ma_r = ma.rotate(angle)
            io.imsave(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_%.3d.png'%(i, angle)), np.array(ma_r))

            ma_f_r = ma_f.rotate(angle)
            io.imsave(os.path.join(train_path, 'groundtruth', 'satImage_%.3d_f_%.3d.png'%(i, angle)), np.array(ma_f_r))

    # Get all images's name
    train_val_images = os.listdir(os.path.join(train_path, 'images'))

    # Split image into train and validation set
    train_images, val_images = train_test_split(train_val_images, test_size=val_size, random_state=seed)

    # Build new folders for validation set
    make_dir(val_path)
    make_dir(os.path.join(val_path, 'images'))
    make_dir(os.path.join(val_path, 'groundtruth'))
    
    # Move validation images to new folders
    for im in val_images:
        os.rename(os.path.join(train_path, 'images', im), os.path.join(val_path, 'images', im))
        os.rename(os.path.join(train_path, 'groundtruth', im), os.path.join(val_path, 'groundtruth', im))


def preprocess_mask(mask):
    """Preprocessing function for masks."""
    mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask
  
def preprocess_img(img):
    """Preprocessing function for images."""
    return img/255

def trainvalGenerator(batch_size, aug_dict, 
                      train_path, val_path,
                      image_folder = 'images', mask_folder = 'groundtruth',
                      train_dir = None, val_dir = None,
                      target_size = (400,400), seed = 1):  
    """Generator for training and validaton set.
    
    Arguments:
        batch_size {int} -- number of images in each batch
        aug_dict {dict} -- dictionary of data augmentation parameters
        train_path {str} -- path of the training set
        val_path {str} -- path of the validation set
        image_folder {str} -- image folder's name (default: {"images"})
        mask_folder {str} -- mask folder's name (default: {"groundtruth"})
        train_dir {str} -- if not None, path to save training set (default: {None})
        val_dir {str} -- if not None, path to save validation set (default: {None})
        target_size {tuple} -- size of targe images (default: {(400,400)})
        seed {int} -- random seed (default: {1})
    
    Returns:
        (trainGen, valGen) -- tuple of genators for training and validation set
    """ 

    # Train
    if train_dir: make_dir(train_dir)

    image_dict = aug_dict.copy()
    image_dict["preprocessing_function"] = preprocess_img
    image_datagen = ImageDataGenerator(**image_dict)
    
    mask_dict = aug_dict.copy()
    mask_dict["preprocessing_function"] = preprocess_mask
    mask_datagen = ImageDataGenerator(**mask_dict)
    
    image_generator_train = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = train_dir,
        save_prefix  = "image",
        seed = seed)
    
    mask_generator_train = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = train_dir,
        save_prefix  = "mask",
        seed = seed)
    
    # Validation
    if val_dir: make_dir(val_dir)

    image_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
    mask_datagen = ImageDataGenerator(preprocessing_function=preprocess_mask)
    
    image_generator_val = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "rgb",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = val_dir,
        save_prefix  = "image",
        seed = seed+1,
        shuffle = False
    )
    
    mask_generator_val = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = val_dir,
        save_prefix  = "mask",
        seed = seed+1,
        shuffle = False
    )
    
    return zip(image_generator_train, mask_generator_train), zip(image_generator_val, mask_generator_val)


def testGenerator(test_path, num_image = 50):
    """Generator for test set.
    
    Arguments:
        test_path {set} -- path of the test set
        num_image {int} -- number of images in the test set (default: {50})
    """
    for i in range(1, num_image+1):
        img = io.imread(os.path.join(test_path, "test_%d"%i, "test_%d.png"%i))
        img = img / 255
        img = np.reshape(img,(1,)+img.shape)
        yield img


def save_result(save_path, npyfile):
    """Save predicted images to path.

    Arguments:
        save_path {str} -- path of the folder to save 
        npyfile {numpy.ndarray} -- numpy array of the predict images
    """
    make_dir(save_path)
    for i, item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path, '%.3d.png'%(i+1)), img)
