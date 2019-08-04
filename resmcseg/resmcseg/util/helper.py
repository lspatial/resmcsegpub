import keras.backend as K
from keras.backend import binary_crossentropy
import numpy as np

color_dict={ 0: (255, 255, 255), # white Background
             1: (0, 0, 0), # Black; Roads
             2: (0, 255, 0),# Green : Grass
             3: (255, 255, 0), # Yellow : Rails
             4: (150, 80, 0), # dark brown : Bare soil
             5: (0, 125, 0), #dark green: trees
             6: (0, 0, 150),# blue water
             7: (100, 100, 100), # gray buildings
             8: (150, 150, 255)} # blue-purple: pools

color_dict_cv2={ 0: (255, 255, 255), # white Background
             1: (0, 0, 0), # Black; Roads
             2: (0, 255, 0),# Green : Grass
             3: (0, 255, 255), # Yellow : Rails
             4: (0, 80, 150), # dark brown : Bare soil
             5: (0, 125, 0), #dark green: trees
             6: (150, 0, 0),# blue water
             7: (100, 100, 100), # gray buildings
             8: (255, 150, 150)} # blue-purple: pools

def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

MAXPIXELVAL=2026.0

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def jaccard_coef1(y_true, y_pred):
    smooth = 1
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

