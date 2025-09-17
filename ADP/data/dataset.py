import glob
import os
import random
import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from ADP.data.aug import *

class ADPDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, batch_size, mask_path=None, augment=True, training=True, test_mask_path=None, world_size=1, trainer=None):
        super(ADPDataset, self).__init__()
        self.config = config
        self.trainer = trainer
        self.augment = augment
        self.training = training
        self.batch_size = batch_size
        self.world_size = world_size
        self.mask_rate = config['mask_rate']
        if training:
            self.input_size = config['input_size']
        else:
            self.eval_size_h = config['eval_size_h']
            self.eval_size_w = config['eval_size_w']

        self.data_instance = []
        self.data = []
        f = open(flist[0], 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()
        f = open(flist[1], 'r')
        for i in f.readlines():
            i = i.strip()
            self.data_instance.append(i)
        f.close()

        if training:
            self.irregular_mask = []
            self.segment_mask = []

            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask.append(line.strip())
            self.irregular_mask = sorted(self.irregular_mask, key=lambda x: x.split('/')[-1])

            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask.append(line.strip())
            self.segment_mask = sorted(self.segment_mask, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

    def __getitem__(self, index):

        img = cv2.imread(self.data[index])
        instance_mask = cv2.imread(self.data_instance[index],cv2.IMREAD_GRAYSCALE)
        instance_mask = (instance_mask > 127).astype(np.uint8) * 255

        while img is None:
            print('image error {}'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])

        img = img[:, :, ::-1]

        size = self.input_size
        if self.training:
            img = self.resize(img, size, size)
        else:
            img = self.resize(img, self.eval_size_h, self.eval_size_w)
        mask = self.load_mask(img, index)

        # augment data
        if self.augment and self.training:
            spatial_transforms = A.Compose([
                IAAPerspective2(scale=(0.0, 0.06),p=0.5,order=0),
                IAAAffine2(scale=(0.7, 1.3),
                           rotate=(-40, 40),
                           shear=(-0.1, 0.1), p=0.5,
                           order=0),
                A.PadIfNeeded(min_height=size, min_width=size,
                              border_mode=cv2.BORDER_REFLECT_101, value=0, mask_value=0),
                A.OpticalDistortion(p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                                    mask_value=0),
                A.RandomCrop(height=size, width=size),
                A.HorizontalFlip(p=0.5),
            ])

            image_only_transforms = A.Compose([
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5, p=0.5),
            ])

            transformed_spatial = spatial_transforms(image=img, mask=instance_mask)
            transformed_img = transformed_spatial['image']
            transformed_instance_mask = transformed_spatial['mask']

            img = image_only_transforms(image=transformed_img)['image']
            instance_mask = transformed_instance_mask
            instance_mask = (instance_mask > 127).astype(np.uint8) * 255

            if random.random() > 0.5:
                mask = mask[:, ::-1, ...].copy()

            if random.random() > 0.5:
                mask = mask[::-1, :, ...].copy()

        batch = dict()
        batch['image'] = F.to_tensor(img).float()
        batch['instance_mask'] = F.to_tensor(instance_mask).float()
        batch['mask'] = F.to_tensor(mask).float()
        batch['name'] = os.path.basename(self.data[index])

        return batch

    def __len__(self):
        return len(self.data)


    def resize(self, img, height, width):
        img_h, img_w = img.shape[0:2]
        if img_h > height and img_w > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_mask(self, img, index):
        img_h, img_w = img.shape[0:2]

        if self.training is False:

            mask = cv2.imread(self.mask_list[index % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask

        else:
            randomnum = random.random()

            if randomnum < self.mask_rate[0]:
                mask_index = random.randint(0, len(self.irregular_mask) - 1)
                mask = cv2.imread(self.irregular_mask[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif randomnum < self.mask_rate[1]:
                mask_index = random.randint(0, len(self.segment_mask) - 1)
                mask = cv2.imread(self.segment_mask[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask) - 1)
                mask1 = cv2.imread(self.segment_mask[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask2 = cv2.imread(self.irregular_mask[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != img_h or mask.shape[1] != img_w:
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255

            return mask

