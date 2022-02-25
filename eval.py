from __future__ import division

import os
import numpy as np
from torch.autograd import Variable
from torch.utils import data as data_
import torch
import random
import ipdb
import argparse
import cv2
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score

from src.head_detector_vgg16 import Head_Detector_VGG16
#from trainer import Head_Detector_Trainer
from src.head_backbone import mpsn
from train_or import Head_Detector_Trainer

from src.config import opt
import src.utils as utils
from data.dataset import Dataset, inverse_normalize
import src.array_tool as at
from src.vis_tool import visdom_bbox
from src.bbox_tools import bbox_iou
import matplotlib.pyplot as plt

dataset_name = 'RGBdata'
phases = ['train', 'val', 'test']
data_check_flag = False

def eval(dataloader, head_detector):
    trainer = Head_Detector_Trainer(head_detector)
    """
    Given the dataloader of the test split compute the
    average corLoc of the dataset using the head detector 
    model given as the argument to the function. 
    """
    test_img_num = 0
    test_corrLoc = 0.0
    AP = 0.0

    for ii, (img_path, img, img2, bbox_, scale) in enumerate(dataloader):

        # print(scale)
        img, bbox = img.cuda().float(), bbox_.cuda()
        img2 = img2.cuda()
        scale = at.scalar(scale)
        # img, bbox = img.cuda().float(), bbox_.cuda()
        img, img2, bbox = Variable(img), Variable(img2), Variable(bbox)

        for gg in range(1):
            pred_bboxes_, scores = head_detector.predict(img, img2, scale, mode='evaluate',
                                                                  thresh=0.000)  # 0。0005
        # s2 = datetime.datetime.now()
        # print(s1,s2)
        # print((s1 - s2).seconds)
        gt_bboxs = at.tonumpy(bbox_)[0]
        pred_bboxes_ = at.tonumpy(pred_bboxes_)





        if pred_bboxes_.shape[0] == 0:
            test_img_num += 1
            continue
        else:
            ious = bbox_iou(pred_bboxes_, gt_bboxs)
            max_ious = ious.max(axis=1)
            corr_preds = np.where(max_ious >= 0.5)[0]
            num_boxs = gt_bboxs.shape[0]
            num_corr_preds = len(corr_preds)

            gt_labels = np.zeros(len(scores))
            for index in corr_preds:
                gt_labels[index] = 1


            scores = np.append(scores, np.zeros(max(num_boxs - num_corr_preds, 0)))  # .astype(np.float)
            gt_labels = np.append(gt_labels, np.ones(max(num_boxs - num_corr_preds, 0)))  # .astype(np.bool_)
            ap = average_precision_score(gt_labels, scores)
            # curve=precision_recall_curve(gt_labels,scores)
            # recall=recall_score(gt_labels,scores)
            AP = AP + ap
            # print(scores)


            test_corrLoc += num_corr_preds / num_boxs
            test_img_num += 1

        img_path_pre = img_path
    print("AP:" + str(AP / test_img_num))

    return AP / test_img_num


def main(args):
    # Get the dataset
    for phase in phases:
        if phase == 'train':
            if dataset_name == 'brainwash':
                train_data_list_path = os.path.join(opt.brainwash_dataset_root_path, 'brainwash_train.idl')
                train_data_list = utils.get_phase_data_list(train_data_list_path, dataset_name)
            if dataset_name == 'RGBdata':
                train_data_list_path = os.path.join(opt.RGBdata_dataset_root_path, 'RGBdata_train.idl')
                train_data_list = utils.get_phase_data_list(train_data_list_path, dataset_name)
        elif phase == 'val':

            if dataset_name == 'brainwash':
                val_data_list_path = os.path.join(opt.brainwash_dataset_root_path, 'brainwash_val.idl')
                val_data_list = utils.get_phase_data_list(val_data_list_path, dataset_name)
            if dataset_name == 'RGBdata':
                val_data_list_path = os.path.join(opt.RGBdata_dataset_root_path, 'RGBdata_val.idl')
                val_data_list = utils.get_phase_data_list(val_data_list_path, dataset_name)
        elif phase == 'test':
            if dataset_name == 'RGBdata':
                test_data_list_path = os.path.join(opt.RGBdata_dataset_root_path, 'RGBdata_test.idl')
                test_data_list = utils.get_phase_data_list(test_data_list_path, dataset_name)
            if dataset_name == 'brainwash':
                test_data_list_path = os.path.join(opt.brainwash_dataset_root_path, 'brainwash_test.idl')
                test_data_list = utils.get_phase_data_list(test_data_list_path, dataset_name)

    print("Number of images for training: %s" % (len(train_data_list)))
    print("Number of images for val: %s" % (len(val_data_list)))
    print("Number of images for test: %s" % (len(test_data_list)))

    if data_check_flag:
        utils.check_loaded_data(train_data_list[random.randint(1, len(train_data_list))])
        utils.check_loaded_data(val_data_list[random.randint(1, len(val_data_list))])
        utils.check_loaded_data(test_data_list[random.randint(1, len(test_data_list))])

    # Load the train dataset
    #train_dataset = Dataset(train_data_list)
    val_dataset = Dataset(val_data_list)
    test_dataset = Dataset(test_data_list)
    print("Load data.")

    #train_dataloader = data_.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = data_.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataloader = data_.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    # Initialize the head detector.
    #head_detector_mpsn = Head_Detector_VGG16(ratios=[1], anchor_scales=[8, 16])
    head_detector_mpsn = mpsn(ratios=[1], anchor_scales=[2, 4])
    print("model construct completed")


    trainer = Head_Detector_Trainer(head_detector_mpsn).cuda()
    trainer.load(args.model_path)
    #avg_test_CorrLoc = eval(val_dataloader, head_detector_mpsn)
    test_Corr = eval(test_dataloader, head_detector_mpsn)
    print("  test average corrLoc accuracy:\t\t{:.3f}".format(test_Corr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set MPSN', add_help=False)
    parser.add_argument('--model_path', type=str,default='./checkpoints/output/diff resnet DFA')
    args = parser.parse_args()

    main(args)
