import os
import glob
import random

import numpy as np
import cv2
import torch
import torch.utils.data as data

import data.util as util


def crop_center(img):
    y,x,c  = img.shape
    dim = min([x,y])
    startx = (x - dim) // 2 
    starty = (y - dim) // 2
    return img[starty:starty+dim,startx:startx+dim,:]


def imread(path):
    res = cv2.imread(path) 
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) / 255.0
    return res


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()


class Dataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        hrs = glob.glob(os.path.join(opt['dataroot_HR'], "*"))
        self.data = []
        for hr_path in hrs:
            video_name = os.path.split(hr_path)[-1].split(".")[0]
            video_frames_path = glob.glob(os.path.join(opt['dataroot_LR'], video_name, "*"))
            video_frames_path = sorted(video_frames_path)
            for video_frame_path in video_frames_path:
                frame_name = os.path.split(video_frame_path)[-1].split(".")[0]
                self.data += [{"hr_path": hr_path,
                              "lr_path": video_frame_path,
                              "video_name": video_name,
                              "frame_name": frame_name}]

        # calculate resolution
        self.hr_resolution = self.opt['resolution']
        self.lr_resoluiton = self.opt['resolution'] // self.opt['scale']

    def __getitem__(self, index):
        data = self.data[index]
        lr = imread(data['lr_path'])
        assert lr.shape[0] == lr.shape[1], "The input is not square"
        lr = cv2.resize(lr, (self.lr_resoluiton, self.lr_resoluiton))

        hr = imread(data['hr_path'])
        assert hr.shape[0] >= 1024
        assert hr.shape[1] >= 1024
        hr = crop_center(hr)
        hr = cv2.resize(hr, (self.hr_resolution, self.hr_resolution))

        res = {'LQ': to_tensor(lr),
               'LQ_path': data['lr_path'],
               "video_name": data['video_name'],
               "frame_name": data['frame_name'],
               "GT_path": ""}
        if self.opt['use_HR_ref']:
            res["img_reference"] = to_tensor(hr)
        return res

    def __len__(self):
        return len(self.data)
