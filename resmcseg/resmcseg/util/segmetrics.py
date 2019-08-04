import numpy as np
from sklearn.metrics import confusion_matrix

import keras.backend as K
from keras.backend import binary_crossentropy
import tensorflow as tf
from resmcseg._metrics import *

def k_nji(y_true, y_pred):
    """
    Function of normal Jaccard Index (keras)
       :param y_true: ground truth mask ;
       :param y_pred: predicted mask  ;
       :return: value of normal JI
    """
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def k_nji_loss(y_true, y_pred):
    """
    Function of log loss based on normal Jaccard Index (keras)
       :param y_true: ground truth mask ;
       :param y_pred: predicted mask  ;
       :return: loss fucntion value
    """
    return -K.log(k_nji(y_true, y_pred))

def k_njibce_loss(y_true, y_pred):
    """
    Function of log loss based on normal Jaccard Index and binary entropy  (keras)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: loss fucntion value
    """
    return -K.log(k_nji(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def k_ji(y_true, y_pred):
    """
    Function of -non-normal Jaccard Index  (keras)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: JI value
    """
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def k_dice_loss(y_true, y_pred):
    """
    Function of dice loss  (keras)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: loss fucntion value
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def k_bce_logdice_loss(y_true, y_pred):
    """
    Function of dice and binary corss entropy loss  (keras)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: loss fucntion value
    """
    return binary_crossentropy(y_true, y_pred) - K.log(1. - K_dice_loss(y_true, y_pred))

def compute_iou(y_pred, y_true,labels=[0,1]):
    """
    mean IOU calculated based on ground truth and predicted values  (np)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: mean IOU
    """
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=labels)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def compute_iou_conf(current):
    """
    IOU calculated based on ground truth and predicted values  (np)
      :param y_true: ground truth mask ;
      :param y_pred: predicted mask  ;
      :return: mean IOU and each class's IOU
    """
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return (np.mean(IoU),IoU)

def jaccard(x, y):
    x = np.asarray(x, np.bool)  # Not necessary, if you keep your data
    y = np.asarray(y, np.bool)  # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def mean_iouC(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 9)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
    y_true: the expected y values as a one-hot
    y_pred: the predicted y values as a one-hot or softmax output
    label: the label to return the IoU for
    Returns:
    the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def miou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
    y_true: the expected y values as a one-hot
    y_pred: the predicted y values as a one-hot or softmax output
    Returns:
    the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        mean_iou = mean_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels