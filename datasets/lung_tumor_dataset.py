import os
import numpy as np
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch
from torch.utils.data import DataLoader
import cv2

class LungTumorDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, masks_dir, data_type, transform=None):
        self.data_dir = data_dir
        self.masks_dir = masks_dir
        self.all_file_names = os.listdir(self.data_dir)
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.all_file_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.all_file_names[idx])
        mask_path = os.path.join(self.masks_dir, self.all_file_names[idx])
        data = np.load(data_path)
        mask = np.load(mask_path)
        
        data = np.moveaxis(data, -1, 0)
        mask = np.expand_dims(mask, axis=0)

        self.save_as_jpg(data, mask, idx)

        return data, mask

    def save_as_jpg(self, data, mask, idx):
        save_dir = f"final/jpg/{self.data_type}/"
        os.makedirs(save_dir, exist_ok=True)

        data_image = np.moveaxis(data, 0, -1)
        data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image)) * 255
        data_image = data_image.astype(np.uint8)
        data_image_path = os.path.join(save_dir, f"{idx}_data.jpg")
        cv2.imwrite(data_image_path, data_image)

        mask_image = mask.squeeze(0)

        unique_values = np.unique(mask_image)

        num_classes = np.max(mask_image) + 1
        if num_classes > 1:
            mask_image = (mask_image / (num_classes - 1) * 255).astype(np.uint8)
        else:
            mask_image = mask_image.astype(np.uint8)

        mask_image_path = os.path.join(save_dir, f"{idx}_mask.jpg")
        cv2.imwrite(mask_image_path, mask_image)

def get_dataset(preprocessed_input_dir, aug_pipeline=None, data_type='train'):
    data_dir = os.path.join(preprocessed_input_dir, data_type, 'data')
    label_dir = os.path.join(preprocessed_input_dir, data_type, 'mask')
    return LungTumorDataset(data_dir, label_dir, data_type, transform=aug_pipeline)