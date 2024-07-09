import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import json
import os

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

# IOU（Intersection over Union）
def get_iou(mask_name, predict):
    image_mask = cv2.imread(mask_name, 0)
    if image_mask is None:
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]

    max_predict = np.max(predict, axis=0)
    predict_2d = np.argmax(predict, axis=0)
    predict_2d[max_predict < 0.2] = 4
    predict = predict_2d.astype(np.int16)
    
    labels = [0, 1, 2, 4]
    iou = 0
    for label in labels:
        interArea = np.sum(np.multiply(predict == label, image_mask == label))
        unionArea = np.sum(np.logical_or(predict == label, image_mask == label))
        iou += interArea / unionArea

    iou = iou/4 # mean iou
    print('Mean IOU: %f' % (iou))
    return iou

# Dice-Sørensen coefficient
def get_dice(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]

    max_predict = np.max(predict, axis=0)
    predict_2d = np.argmax(predict, axis=0)
    predict_2d[max_predict < 0.2] = 4
    predict = predict_2d.astype(np.int16)

    labels = [0, 1, 2, 4]
    dice = 0
    for label in labels:
        interArea = np.sum(np.multiply(predict == label, image_mask == label))
        predictArea = np.sum(predict == label)
        maskArea = np.sum(image_mask == label)
        dice += (2. * interArea) / (predictArea + maskArea)

    dice = dice/4 # mean dice
    print('Dice-Sørensen coefficient: %f' % (dice))
    return dice
    

# Hausdorff Distance
def get_hd(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]

    max_predict = np.max(predict, axis=0)
    predict_2d = np.argmax(predict, axis=0)
    predict_2d[max_predict < 0.2] = 4

    mask_points = np.column_stack(np.where((image_mask != 4)))
    predict_points = np.column_stack(np.where(predict_2d != 4))

    # res = np.sum(predict != image_mask)

    hd1 = directed_hausdorff(mask_points, predict_points)[0]
    hd2 = directed_hausdorff(predict_points, mask_points)[0]
    res = max(hd1, hd2)
    print('Hausdorff Distance of: %f' % (res))
    
    return res



def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()
