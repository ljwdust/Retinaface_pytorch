#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/8 3:52 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com

import os
import tqdm
import pickle
import argparse
import numpy as np
import cv2
from bbox import bbox_overlaps
import pickle

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""
    cache_file = os.path.join(gt_dir, 'gt_box.cache')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    gt_dict = {}
    for i in range(1, 11):
        filename = os.path.join(gt_dir, 'FDDB-fold-{}-ellipseList.txt'.format('%02d' % i))
        assert os.path.exists(filename)
        gt_sub_dict = {}
        annotationfile = open(filename)
        while True:
            filename = annotationfile.readline()[:-1].replace('/', '_')
            if not filename:
                break
            line = annotationfile.readline()
            if not line:
                break
            facenum = int(line)
            face_loc = []
            for j in range(facenum):
                line = annotationfile.readline().strip().split()
                major_axis_radius = float(line[0])
                minor_axis_radius = float(line[1])
                angle = float(line[2])
                center_x = float(line[3])
                center_y = float(line[4])
                score = float(line[5])
                angle = angle / 3.1415926 * 180
                mask = np.zeros((1000, 1000), dtype=np.uint8)
                cv2.ellipse(mask, ((int)(center_x), (int)(center_y)),
                            ((int)(major_axis_radius), (int)(minor_axis_radius)), angle, 0., 360., (255, 255, 255))
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
                r = cv2.boundingRect(contours[0])
                x_min = r[0]
                y_min = r[1]
                x_max = r[0] + r[2]
                y_max = r[1] + r[3]
                face_loc.append([x_min, y_min, x_max, y_max])
            face_loc = np.array(face_loc)

            gt_sub_dict[filename] = face_loc
        gt_dict[i] = gt_sub_dict

    with open(cache_file, 'wb') as f:
        pickle.dump(gt_dict, f, pickle.HIGHEST_PROTOCOL)

    return gt_dict


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))

    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    pred = get_preds(pred)
    #norm_score(pred)
    gt_box_dict = get_gt_boxes(gt_path)
    event = list(pred.keys())
    event = [int(e) for e in event]
    event.sort()
    thresh_num = 1000
    aps = []

    pbar = tqdm.tqdm(range(len(event)))
    for setting_id in pbar:
        pbar.set_description('Predicting ... ')
        # different setting
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')

        gt = gt_box_dict[event[setting_id]]
        pred_list = pred[str(event[setting_id])]
        gt_list = list(gt.keys())
        for j in range(len(gt_list)):
            gt_boxes = gt[gt_list[j]].astype('float')  # from image name get gt boxes
            pred_info = pred_list[gt_list[j]]
            keep_index = np.array(range(1, len(gt_boxes) + 1))
            count_face += len(keep_index)
            ignore = np.zeros(gt_boxes.shape[0])
            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            if len(keep_index) != 0:
                ignore[keep_index - 1] = 1
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

            pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    for i in range(len(aps)):
        print("FDDB-fold-{} Val AP: {}".format(event[i], aps[i]))
    print("FDDB Dataset Average AP: {}".format(sum(aps)/len(aps)))
    print("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./fddb_evaluation/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)
