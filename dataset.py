import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join
import itertools
import glob
import cv2
from albumentations import (Compose,
                            Affine,
                            ElasticTransform,
                            GridDistortion,
                            RandomBrightnessContrast,
                            HueSaturationValue,
                            RandomGamma,
                            RandomCrop,
                            RandomRotate90,
                            RandomScale,
                            ChannelShuffle,
                            ColorJitter,
                            PadIfNeeded
                            )



def get_ok_transform(size=(512, 512)):
    aug = Compose([
        Affine(translate_percent=(0.1, 0.1), rotate=(-10, 10), scale=(0.9, 1.1), shear=(-10, 10), p=0.5, mode=cv2.BORDER_REFLECT_101),
        ElasticTransform(alpha=1, sigma=50 * 0.05, alpha_affine=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
    ], p=1)
    return aug


def get_defect_transform(size=(512, 512)):
    aug = Compose([
        Affine(translate_percent=(0.1, 0.1), rotate=(-10, 10), scale=(0.9, 1.1), shear=(-10, 10), p=0.5, mode=cv2.BORDER_REFLECT_101),
        ElasticTransform(alpha=1, sigma=50 * 0.05, alpha_affine=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        RandomScale(scale_limit=0.2, p=0.5),
        PadIfNeeded(min_height=size[1], min_width=size[0], p=0.5, border_mode=cv2.BORDER_REFLECT_101, always_apply=True),
        RandomCrop(height=size[1], width=size[0], always_apply=True),
    ], p=0.6)
    return aug


class DefusionDataset(Data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self,
                 src_path: str,
                 targrt_path: str,
                 size=(1024, 768),
                 ok_transform=None,
                 defect_transform=None
                 ):
        """

        Args:
            src_path (str): ok fold
            targrt_path (str): _description_
            size (tuple, optional): _description_. Defaults to (1024, 768).
            transform (_type_, optional): _description_. Defaults to None.
        """
        # 'Initialization'
        super(DefusionDataset, self).__init__()
        self.data_path = src_path
        self.targrt_path = targrt_path
        
        self.size = size
        self.ok_transform = ok_transform
        self.defect_transform = defect_transform
        
        self.src_files = [f for f in glob.glob(os.path.join(self.data_path, '*.png')) if 'mask' not in os.path.basename(f)]
        self.target_files = [f for f in glob.glob(os.path.join(self.targrt_path, 'train/*.png')) if 'mask' not in os.path.basename(f)]
        self.target_mask_files = [f for f in glob.glob(os.path.join(self.targrt_path, 'trainannot/*.png'))]
        
        self.target_files.sort()
        self.target_mask_files.sort()
        
    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.src_files)

    def totensor(self, *images):
        return [torch.from_numpy(np.transpose(img, (2, 0, 1))) for img in images]

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        img_p = self.src_files[index]
        idx = np.random.randint(0, len(self.target_files))
        target_p = self.target_files[idx]
        pos_mask_p = self.target_mask_files[idx]
        
        img, img_target, pos_mask = cv2.imread(img_p), cv2.imread(target_p), cv2.imread(pos_mask_p)
        pos_mask = np.where(pos_mask > 0 , 255., 0.)
        img, img_target = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
        
        img, img_target, pos_mask = img.astype(np.float32), img_target.astype(np.float32), pos_mask.astype(np.float32)
        img, img_target, pos_mask = img / 255.0, img_target / 255.0, pos_mask / 255.0
        
        img, img_target, pos_mask = [cv2.resize(img, self.size),
                                     cv2.resize(img_target, self.size),
                                     cv2.resize(pos_mask, self.size, interpolation=cv2.INTER_NEAREST)]

                
        if self.ok_transform is not None:
            img = self.ok_transform(image=img)['image']
        
        if self.defect_transform is not None:
            sample = self.defect_transform(image=img_target, mask=pos_mask)
            img_target, pos_mask = sample['image'], sample['mask']
            
        pos_src = img * pos_mask
        pos_dst = img_target * pos_mask
        fused = img * (1 - pos_mask) + pos_dst
        normal_mask = (1 - pos_mask)
        
        img, pos_dst, img_target, pos_mask, fused, normal_mask = self.totensor(img, pos_dst, img_target, pos_mask, fused, normal_mask)
        
        return img, pos_dst, img_target, pos_mask, fused, normal_mask