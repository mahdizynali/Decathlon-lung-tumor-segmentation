import numpy as np
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import cv2
import torch
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa

class Augmenter:
    def __init__(self, transform):
        self.transform = transform

    def augment(self, data, mask):
        random_seed = torch.randint(0, 1000000, (1,)).item()
        ia.seed(random_seed)
        mask = SegmentationMapsOnImage(mask, shape=mask.shape)
        aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
        return aug_data, aug_mask.get_arr()

def save_data_and_mask(data, mask, output_path_prefix, name, size, vgg_compatible=True):
    resized_data = cv2.resize(data, size).astype(np.float32)
    if vgg_compatible:
        resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    np.save(os.path.join(output_path_prefix, 'data', name), resized_data)
    np.save(os.path.join(output_path_prefix, 'mask', name), resized_mask)

def preprocess_ct_scan(img_path, label_path, output_path_prefix, f_name, size,
                       frames_to_skip=30, orientation=('L', 'A', 'S'), vgg_compatible=True,
                       scaling_value=3071, augmenter=None):
    scan_data = nib.load(img_path)
    ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
    label_volume = nib.load(label_path).get_fdata().astype(np.uint8)
    
    os.makedirs(os.path.join(output_path_prefix, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_path_prefix, 'mask'), exist_ok=True)
    
    for idx in range(frames_to_skip, ct_scan_volume.shape[-1]):
        if nib.aff2axcodes(scan_data.affine) == orientation:
            name = f'{f_name}_{idx}'

            save_data_and_mask(ct_scan_volume[:, :, idx], label_volume[:, :, idx], output_path_prefix, name, size, vgg_compatible)

            if augmenter:
                aug_data, aug_mask = augmenter.augment(ct_scan_volume[:, :, idx], label_volume[:, :, idx])
                save_data_and_mask(aug_data, aug_mask, output_path_prefix, f"{name}_aug", size, vgg_compatible)
        else:
            print(f"{f_name} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")

def preprocess(input_data_dir, input_labels_dir, output_dir, frames_to_skip, size, orientation, vgg_compatible,
               val_split=0.1, augment=False):
    valid_files = [scan_file for scan_file in os.listdir(input_data_dir) if scan_file.startswith('lung') and
                   scan_file.endswith('.nii.gz')]
    train_size = len(valid_files) * (1 - val_split)

    if augment:
        # transform = iaa.Sequential([
        #     iaa.Fliplr(0.5),
        #     iaa.Flipud(0.5),
        #     iaa.LinearContrast((0.75, 1.25)),
        #     iaa.Affine(translate_percent=0.15, scale=(0.85, 1.15), rotate=(-45, 45)),
        #     iaa.ElasticTransformation()
        # ])
        transform = iaa.Sequential([
            iaa.Fliplr(0.5),  # Horizontal flip
            iaa.Flipud(0.5),  # Vertical flip
            iaa.Affine(rotate=(-10, 10)),  # Small rotations
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),  # Random translation
            iaa.Affine(scale=(0.9, 1.1)),  # Scaling
            iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=0.1),  # Elastic deformation
            iaa.LinearContrast((0.75, 1.25)),  # Contrast adjustment
            iaa.CropAndPad(px=(0, 10), pad_mode="constant", pad_cval=0),  # Random cropping and padding
            iaa.GaussianBlur(sigma=(0.0, 1.0)),  # Gaussian blur
            iaa.Sharpen(alpha=(0.1, 0.5), lightness=(0.75, 1.25))  # Sharpening
        ])
    else:
        transform = None

    for idx, scan_file in tqdm(enumerate(valid_files)):
        output_prefix = 'train' if idx < train_size else 'val'
        output_path_prefix = os.path.join(output_dir, output_prefix)
        img_path = os.path.join(input_data_dir, scan_file)
        label_path = os.path.join(input_labels_dir, scan_file)
        f_name = scan_file.split(".")[0]
        print(" ",f_name)

        if output_prefix == 'train' and transform:
            augmenter = Augmenter(transform=transform)
        else:
            augmenter = None

        with ProcessPoolExecutor() as executor:
            executor.submit(preprocess_ct_scan, img_path, label_path, output_path_prefix, f_name, size, frames_to_skip,
                            orientation, vgg_compatible, augmenter=augmenter)

preprocess(input_data_dir="../../dataset/imagesTr", 
            input_labels_dir="../../dataset/labelsTr/", 
            output_dir="final", 
            frames_to_skip=30, 
            size=(224, 224),
            orientation=('L', 'A', 'S'), 
            vgg_compatible=True, 
            val_split=0.1, 
            augment=True)