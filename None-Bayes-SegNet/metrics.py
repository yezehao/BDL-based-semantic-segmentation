import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import json
import os
from PIL import Image

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
    # print('Mean IOU: %f' % (iou))
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
    # print('Dice-Sørensen coefficient: %f' % (dice))
    return dice

def get_precision(mask_name, predict): # Precision / Recall / F1
    image_mask = cv2.imread(mask_name, 0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]

    max_predict = np.max(predict, axis=0)
    predict_2d = np.argmax(predict, axis=0)
    predict_2d[max_predict < 0.6] = 4
    predict = predict_2d.astype(np.int16)

    # True Positives (TPs)
    TP =  np.sum(np.multiply(predict == 0, image_mask == 0))
    # False Positives (FPs)
    FP = np.sum(np.multiply(predict == 0, image_mask != 0))
    # False Negatives (FNs)
    FN = np.sum(np.multiply(predict != 0, image_mask == 0))
    Pr = 0 if (TP + FP) == 0 else TP/(TP + FP) # Precision (Pr)
    Re = 0 if (TP + FN) == 0 else TP/(TP + FN) # Recall (Re)
    F1 = 0 if (Pr + Re) == 0 else 2*Pr*Re/(Pr + Re) # Harmonic Mean F1

    return Pr, Re, F1
    

def test_show(root, threshold, img_number, prediction):
    # Path
    jpg_path = os.path.join(root, f'test/{img_number}.jpg')
    GT_path = os.path.join(root, f'test_mask/{img_number}m.png') # Groud Truth

    # Define Colour
    colors = {
        0: (247, 195, 37, 128),    # yellow (RGBA) = obstacle & environment
        1: (41, 167, 224, 128),    # cyan-blue (RGBA) = water
        2: (90, 75, 164, 128),     # purple (RGBA) = sky
        3: (255, 0, 0, 128)        # red (RGBA) = unknown
    }

    # Original JPG Image
    image = Image.open(jpg_path).convert('RGBA')
    # Prediction Segmentation
    mask_array = np.array(prediction)
    mask_array = np.transpose(mask_array, (1, 2, 0))
    mask_indices = np.argmax(mask_array, axis=2)
    mask_values = np.max(mask_array, axis=2)
    mask_indices[mask_values < threshold] = 3


    mask_image = Image.new('RGBA', image.size)
    mask_pixels = mask_image.load()
    for y in range(mask_indices.shape[0]):
        for x in range(mask_indices.shape[1]):
            value = mask_indices[y, x]
            if value in colors:
                mask_pixels[x, y] = colors[value]

    # Ground Truth
    GT_mask = Image.open(GT_path)
    GT = np.array(GT_mask)
    GT[GT==4] = 3

    mask_gt = Image.new('RGBA', image.size)
    gt_pixels = mask_gt.load()
    for y in range(GT.shape[0]):
        for x in range(GT.shape[1]):
            value = GT[y, x]
            if value in colors:
                gt_pixels[x, y] = colors[value]

    # apply mask to jpg
    combined_pred = Image.alpha_composite(image, mask_image)
    combined_gt = Image.alpha_composite(image, mask_gt)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Predicted segmentation
    plt.subplot(1, 2, 1)
    plt.imshow(combined_pred)
    plt.title(f'Predicted Segmentation')
    plt.axis('off')

    # Ground truth segmentation
    plt.subplot(1, 2, 2)
    plt.imshow(combined_gt)
    plt.title(f'Ground Truth Segmentation')
    plt.axis('off')

    plt.tight_layout()
    plot_path = f"result/testing/{img_number}.png"
    plt.savefig(plot_path)

    print(f"Testing Image: {img_number}")

def loss_plot(args,loss):
    num = args.epoch
    x = [i for i in range(num)]
    plot_save_path = r'result/training'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path+'loss_'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.jpg'
    plt.figure()
    plt.plot(x,loss,label='loss')
    plt.legend()
    plt.savefig(save_loss)
