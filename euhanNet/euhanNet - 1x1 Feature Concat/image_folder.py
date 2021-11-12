import torch.utils.data as data
from PIL import Image
import glob
import os
import os.path as osp
import cv2
from scipy.io import loadmat
import numpy as np


def make_dataset(directory, opt):
    img = glob.glob(osp.join(directory, 'Images_3x3/*'))

    return img


class Image_Folder(data.Dataset):
    def __init__(self, opt, root, transform):

        self.opt = opt
        self.root = root
        self.imgs = make_dataset(root, opt)
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.imgs[index]
        name = img_path
        name = name.split("\\")
        name1 = name[1]
        name1 = name1[0:-4]
        img3x3 = Image.open(img_path)
        img3x3 = img3x3.resize((300,300))
        img1x1 = Image.open(img_path.replace("3x3", "1x1"))
        img1x1 = img1x1.resize((100,100))
        gt = loadmat(img_path.replace('Images_3x3','Ground_truth_heatmap').replace("png", "mat"))
        gt = gt['map1']
        gt = cv2.resize(gt, (300,300))

        #cv2.imwrite(os.path.join('euhanNet_1x1 Feature Concat_results/gt_test', name1 + '.png'), gt)
        #gt = np.expand_dims(gt, axis = 0)

        img3x3 = self.transform(img3x3)
        img1x1 = self.transform(img1x1)

        return img3x3, img1x1, gt, name[-1]


    def __len__(self):

        return len(self.imgs)
