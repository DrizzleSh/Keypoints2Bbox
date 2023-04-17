#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2023/4/10 14:46
# @Author   : DrizzleSh
# @Usage    : use the SAM
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3)], np.array([0.6]), axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]  # 取出宽高
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    return mask_image


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    # 画点 [x0, x1, x2 ......] [y0, y1, y2 ......]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)     # 面积从大到小排序
    ax = plt.gca()  # get current axes
    ax.set_autoscale_on(False)  # 设置是否在绘图命令上应用自动缩放
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.25)))   # 用颜色显示图像

from torch.utils.data import Dataset, DataLoader
class PtnDataset(Dataset):
    # def __new__(cls, path_dir: str, device):
    #     return super(PtnDataset, cls).__new__(cls)
    def __init__(self, path_dir: str):
        self.image_list = []
        self.anno_list = []
        pbar = tqdm(sorted(os.listdir(path_dir)))
        for file_name in pbar:
            total_path = os.path.join(path_dir, file_name)
            if file_name.endswith('.jpg'):
                self.image_list.append(total_path)
            else:
                self.anno_list.append(total_path)
        assert len(self.anno_list) == len(self.image_list), '图片和标签数量不对应'

    def __getitem__(self, item):
        image_path, anno_path = self.image_list[item], self.anno_list[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(self.anno_list[item], 'r') as f:
            json_data = json.load(f)
        labels = []
        label = []
        is_firsttime = True
        # cur_label = '0'
        for i, shape in enumerate(json_data['shapes']):
            if is_firsttime:    # 第一次进来定义一下cur_label
                is_firsttime = False
                cur_label = shape['label']
            # if shape['shape_type'] == 'point':
            if shape['shape_type'] == 'rectangle':
                # box = shape['points'][0] + shape['points'][1]
                # labels.append(box)
                continue
            if shape['label'] != cur_label:
                label = self.kpt2box(label, cur_label)
                labels.append(label)    # 每换一个人，添加一组数据上去
                label = []
                cur_label = shape['label']

            group_id = int(shape['group_id'])
            # if 0 < group_id < 5:
            #     continue

            label += shape['points']

        label = self.kpt2box(label, cur_label)
        labels.append(label)    # 加上最后一组


            # box = shape['points'][0] + shape['points'][1]
            # label.append(box)
        # image.to(device=device)
        # label.to(device=device)=
        return image, labels

    def __len__(self):
        return len(self.image_list)

    def kpt2box(self, label, cur_label):
        label = np.array(label)
        label = np.transpose(label)
        xmin, xmax = np.min(label[0]) - 20, np.max(label[0]) + 20
        ymin, ymax = np.min(label[1]) - 20, np.max(label[1]) + 20
        label = label.transpose().tolist()
        label.append([xmin, ymin, xmax, ymax, cur_label])
        return label

def collate_fn(data):
    img, labs = data[0]
    # for i, lab in enumerate(labs):
    #     labs[i] = np.array(lab)
    for i, person in enumerate(labs):
        # for j, point in enumerate(person):
        labs[i] = [np.array(person[:-1]), np.array(person[-1])]
    return img, labs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_ckpt', type=str, default='./default.pth', help='sam_checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='GPU or CPU')
    parser.add_argument('--model_type', type=str, default='default', help='model_type')
    parser.add_argument('--images_dir', type=str, default='', help='Dir of images')
    # parser.add_argument('--result_dir', type=str, default='', help='Dir to save results')
    args = parser.parse_args()

    sam_checkpoint = args.sam_ckpt
    device = args.device
    model_type = args.model_type
    images_dir = args.images_dir
    # result_dir = args.result_dir

    print('loading model......')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generate = SamPredictor(sam)

    print('loading dataset......')
    dataset = PtnDataset(images_dir)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            num_workers=8,
                            collate_fn=collate_fn)

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader))
    for i, (image, label) in pbar:  # uint8
        pbar.set_description(f'正在处理第{i}幅图片')

        filename = str(i).zfill(6) + '.json'
        file_path = os.path.join(images_dir, filename)
        with open(file_path, 'r') as f:
            datas = json.load(f)

        mask_generate.set_image(image)
        # newlabels = []
        for points in label:   # 每次更新一个框(人)
            # point_labels = np.ones(len(points[0]))
            mask, _, _ = mask_generate.predict(box=points[1][:-1],
                                               # point_coords=points[0],
                                               # point_labels=point_labels,
                                               multimask_output=False)   # 返回所有的分割对象
            cv_mask = (mask[0] ^ False).astype(np.uint8)
            retval, _, stats, _ = cv2.connectedComponentsWithStats(cv_mask, 8)
            h, w = cv_mask.shape
            xmin, xmax, ymin, ymax = w, 0, h, 0
            for stat in stats[1:]:
                x1, y1, x2, y2 = stat[0], stat[1], stat[0] + stat[2], stat[1] + stat[3]
                xmin = x1 if x1 < xmin else xmin
                xmax = x2 if x2 > xmax else xmax
                ymin = y1 if y1 < ymin else ymin
                ymax = y2 if y2 > ymax else ymax
            newlabel = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
            # newlabels.append(newlabel)
            # 写json文件
            shape = {}
            shape['label'] = points[1][-1]
            shape['points'] = newlabel
            shape['group_id'] = 'None'
            shape['shape_type'] = 'rectangle'
            shape['flags'] = {}
            datas['shapes'].append(shape)

        with open(file_path, 'w') as f:
            json.dump(datas, f)


        # for nlabel in newlabels:
        #     cv2.rectangle(image, (nlabel[0], nlabel[1]), (nlabel[2], nlabel[3]), (255, 0, 0), 2)
        # filename = str(i).zfill(6) + '.jpg'
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(result_dir, filename), image)
