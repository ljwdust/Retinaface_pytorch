# widerface训练集，加入了旋转图片和无人脸图片数据
import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)
        
        # 读取无人脸图像列表
        non_path_txt = './data/no_face.txt'
        with open(non_path_txt, 'r', encoding='utf-8') as f_non:
            non_list = f_non.readlines()
        for path in non_list:
            self.imgs_path.append(path.strip())
            self.words.append([])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            # 若无人脸，设置标签
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = 0  # x1
            annotation[0, 1] = 0  # y1
            annotation[0, 2] = 0  # x2
            annotation[0, 3] = 0  # y2

            # landmarks
            annotation[0, 4] = -1  # l0_x
            annotation[0, 5] = -1   # l0_y
            annotation[0, 6] = -1   # l1_x
            annotation[0, 7] = -1   # l1_y
            annotation[0, 8] = -1   # l2_x
            annotation[0, 9] = -1   # l2_y
            annotation[0, 10] = -1   # l3_x
            annotation[0, 11] = -1   # l3_y
            annotation[0, 12] = -1   # l4_x
            annotation[0, 13] = -1   # l4_y
            annotation[0, 14] = 0
            
            annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)
        else:
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

                annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)
            # 随机旋转图片
            img, target = rotate_image_and_points(img, target, [0, 90, 180, 270], [0.7, 0.1, 0.1, 0.1])

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def rotate_image_and_points(image, data_matrix, angles, probabilities):
    """
    随机旋转图片并调整多个目标的人脸框和关键点坐标。
    :param image: 输入的图像 (numpy array)
    :param data_matrix: 人脸数据矩阵，每行格式 [x1, y1, x2, y2, x1_lmk, y1_lmk, ..., x5_lmk, y5_lmk]
    :param angles: 可选旋转角度列表 [0, 90, 180, 270]
    :param probabilities: 对应的概率分布，长度与 angles 相同
    :return: 旋转后的图像、调整后的人脸数据矩阵
    """
    h, w = image.shape[:2]
    angle = random.choices(angles, probabilities)[0]
    rotated_image = image.copy()

    new_data_matrix = data_matrix.copy() # 创建一个新的副本

    if angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_data_matrix[:, [0, 2]] = h - data_matrix[:, [3, 1]]  # bbox x 调整
        new_data_matrix[:, [1, 3]] = data_matrix[:, [0, 2]]  # bbox y 调整
        new_data_matrix[:, 4:-1:2] = h - data_matrix[:, 5:-1:2]  # landmarks 旋转
        new_data_matrix[:, 5:-1:2] = data_matrix[:, 4:-1:2]

    elif angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        new_data_matrix[:, [0, 2]] = w - data_matrix[:, [2, 0]]
        new_data_matrix[:, [1, 3]] = h - data_matrix[:, [3, 1]]
        new_data_matrix[:, 4:-1:2] = w - data_matrix[:, 4:-1:2]
        new_data_matrix[:, 5:-1:2] = h - data_matrix[:, 5:-1:2]

    elif angle == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_data_matrix[:, [0, 2]] = data_matrix[:, [1, 3]]
        new_data_matrix[:, [1, 3]] = w - data_matrix[:, [2, 0]]
        new_data_matrix[:, 4:-1:2] = data_matrix[:, 5:-1:2]
        new_data_matrix[:, 5:-1:2] = w - data_matrix[:, 4:-1:2]

    return rotated_image, new_data_matrix


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
