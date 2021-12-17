from __future__ import division
import copy
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

from src.head_backbone import mpsn
from train_or import Head_Detector_Trainer
from src.config import opt
import src.utils as utils
from data.dataset import Dataset, inverse_normalize
import src.array_tool as at
from src.vis_tool import visdom_bbox
from src.bbox_tools import bbox_iou
import matplotlib.pyplot as plt

plt.switch_backend('agg')
dataset_name = 'RGBdata'
phases = ['train', 'val', 'test']
data_check_flag = False
torch.cuda.set_device(0)


def save(trainer):
    save_dict = dict()
    save_dict['model'] = trainer.head_detector.state_dict()
    save_dict['config'] = opt._state_dict()
    return save_dict


def eval(dataloader, head_detector):
    trainer = Head_Detector_Trainer(head_detector)

    # trainer.load('./checkpoints/diff_head_detector03120716_0.8')
    # print(os.getcwd())
    #  trainer.load('./checkpoints/diff_head_detector04151658_0.895')
    # trainer.load('./checkpoints/head_detector12200617_0.813')
    # trainer.load('./checkpoints/head_detector12251121_0.809')

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
        # s1 = datetime.datetime.now()
        for gg in range(1):
            pred_bboxes_, scores = head_detector.predict(img, img2, scale, mode='evaluate',
                                                         thresh=0.000)  # 0。0005
        # s2 = datetime.datetime.now()
        # print(s1,s2)
        # print((s1 - s2).seconds)
        gt_bboxs = at.tonumpy(bbox_)[0]
        pred_bboxes_ = at.tonumpy(pred_bboxes_)

        # img2 = img.clone()

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
            # show_img(img_path,img_path_pre,pred_bboxes_,scores,cam_img)

            test_corrLoc += num_corr_preds / num_boxs
            test_img_num += 1
        img_path_pre = img_path
    print("AP:" + str(AP / test_img_num))

    return AP / test_img_num


def train():
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
    train_dataset = Dataset(train_data_list)
    val_dataset = Dataset(val_data_list)
    test_dataset = Dataset(test_data_list)
    print("Load data.")

    train_dataloader = data_.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = data_.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    test_dataloader = data_.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    # Initialize the head detector.
    scales1 = [4,8]
    head_detector_mpsn = mpsn(ratios=[1], anchor_scales=scales1)
    print("model construct completed")

    trainer = Head_Detector_Trainer(head_detector_mpsn).cuda()
    lr_ = opt.lr
    test_ap = []
    val_ap = []
    ep = []
    log = []
    best_val = 0
    best_test = 0

    for epoch in range(opt.epoch):
        log_tmp = []
        # trainer.reset_meters()
        for ii, (img_path, img, img2, bbox_, scale) in enumerate(train_dataloader):
            scale = at.scalar(scale)
            img, bbox = img.cuda().float(), bbox_.cuda()
            img2 = img2.cuda()
            img, img2, bbox = Variable(img), Variable(img2), Variable(bbox)
            _, _, _ = trainer.train_step(img, img2, bbox, scale)
        val_Corr = round(eval(val_dataloader, head_detector_mpsn), 3)
        test_Corr = round(eval(test_dataloader, head_detector_mpsn), 3)
        test_ap.append(test_Corr)
        val_ap.append(val_Corr)
        ep.append(epoch)

        print("Epoch {} of {}.".format(epoch + 1, opt.epoch))
        log_tmp = "Epoch {} of {}.\n".format(epoch + 1, opt.epoch)
        log.append(log_tmp)
        if test_Corr > best_test:
            best_test = test_Corr
            test_model = copy.deepcopy(trainer)
        if val_Corr > best_val:
            best_val = val_Corr
            val_model = copy.deepcopy(trainer)

        print("  test average corrLoc accuracy:\t\t{:.3f}".format(test_Corr))
        print("  val average corrLoc accuracy:\t\t{:.3f}".format(val_Corr))
        log_tmp = "  test average corrLoc accuracy:\t\t{:.3f}\n".format(test_Corr)
        log.append(log_tmp)
        log_tmp = "  val average corrLoc accuracy:\t\t{:.3f}\n".format(val_Corr)
        log.append(log_tmp)
        if epoch == 15 or epoch == 35 or epoch == 42:
            # trainer.load(model_save_path)
            #trainer.head_detector.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay  # 0.1

        # model_save_path = trainer.save(best_map=avg_test_CorrLoc)



    name = 'resnet diff DFA+APC RGBdata '
    title = name + ' Anchor_scales {:} AP '.format(scales1)
    log_tmp = title + 'best_test_ap:\t{:.3f}\n'.format(best_test)
    log.append(log_tmp)
    print(log_tmp)
    log_tmp = title + 'best_val_ap:\t{:.3f}\n'.format(best_val)
    log.append(log_tmp)
    print(log_tmp)


    # plt.figure()
    plt.title(title)
    plt.plot(ep, test_ap, color='red', label='Test')
    plt.plot(ep, val_ap, color='blue', label='Val')
    plt.legend(loc='upper left')
    save_path = title + '.png'
    save_path = os.path.join('output', save_path)
    plt.savefig(save_path)
    txt_path = 'output/' + title + ' .txt'
    with open(txt_path, 'w+') as f:
        f.writelines(log)
    f.close()
    name1 = name + ' test ' + str(best_test)

    model_save_path = os.path.join(opt.model_save_path, name1)
    test_model.save(save_path=model_save_path, best_map=best_test)

    name1 = name + ' val ' + str(best_val)
    model_save_path = os.path.join(opt.model_save_path, name1)
    val_model.save(save_path=model_save_path, best_map=best_val)


if __name__ == "__main__":
    train()
