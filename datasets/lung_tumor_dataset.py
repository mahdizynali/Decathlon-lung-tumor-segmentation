# import os
# import numpy as np
# import imgaug as ia
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# import torch
# from torch.utils.data import DataLoader
# import cv2

# class LungTumorDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, masks_dir, transform=None):
#         self.data_dir = data_dir
#         self.masks_dir = masks_dir
#         self.all_file_names = os.listdir(self.data_dir)
#         self.transform = transform
#         self.save = True
        

#     def __len__(self):
#         return len(self.all_file_names)

#     # def augment(self, data, mask):
#     #     random_seed = torch.randint(0, 1000000, (1,)).item()
#     #     ia.seed(random_seed)

#     #     mask = SegmentationMapsOnImage(mask, shape=mask.shape)
#     #     aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
#     #     return aug_data, aug_mask.get_arr()

#     def augment(self, data, mask):

#         random_seed = torch.randint(0, 1000000, (1,)).item()
#         ia.seed(random_seed)

#         mask = SegmentationMapsOnImage(mask, shape=mask.shape)
#         aug_data, aug_mask = self.transform(image=data, segmentation_maps=mask)
#         return aug_data, aug_mask.get_arr()


#     def __getitem__(self, idx):
#         data_path = os.path.join(self.data_dir, self.all_file_names[idx])
#         mask_path = os.path.join(self.masks_dir, self.all_file_names[idx])
#         data = np.load(data_path)
#         mask = np.load(mask_path)
        
#         if self.transform:
#             aug_data, aug_mask = self.augment(data, mask)
#             aug_data = np.moveaxis(aug_data, -1, 0)
#             aug_mask = np.expand_dims(aug_mask, axis=0)
#         else:
#             aug_data, aug_mask = None, None
        
#         data = np.moveaxis(data, -1, 0)
#         mask = np.expand_dims(mask, axis=0)
        
#         if self.save:
#             self.save_as_jpg(data, mask, aug_data, aug_mask, idx)

#         return (data, mask), (aug_data, aug_mask)

#     def save_as_jpg(self, data, mask, aug_data, aug_mask, idx):
#         save_dir = "final/jpg/"
#         os.makedirs(save_dir, exist_ok=True)

#         data_image = np.moveaxis(data, 0, -1)
#         data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image)) * 255
#         data_image = data_image.astype(np.uint8)
#         data_image_path = os.path.join(save_dir, f"{idx}_data.jpg")
#         cv2.imwrite(data_image_path, data_image)

#         mask_image = mask.squeeze(0)

#         unique_values = np.unique(mask_image)

#         num_classes = np.max(mask_image) + 1
#         if num_classes > 1:
#             mask_image = (mask_image / (num_classes - 1) * 255).astype(np.uint8)
#         else:
#             mask_image = mask_image.astype(np.uint8)

#         mask_image_path = os.path.join(save_dir, f"{idx}_mask.jpg")
#         cv2.imwrite(mask_image_path, mask_image)

#         if aug_data is not None:
#             aug_data_image = np.moveaxis(aug_data, 0, -1)
#             aug_data_image = (aug_data_image - np.min(aug_data_image)) / (np.max(aug_data_image) - np.min(aug_data_image)) * 255
#             aug_data_image = aug_data_image.astype(np.uint8)
#             aug_data_image_path = os.path.join(save_dir, f"{idx}_aug_data.jpg")
#             cv2.imwrite(aug_data_image_path, aug_data_image)

#             aug_mask_image = aug_mask.squeeze(0)

#             unique_values = np.unique(aug_mask_image)

#             num_classes = np.max(aug_mask_image) + 1
#             if num_classes > 1:
#                 aug_mask_image = (aug_mask_image / (num_classes - 1) * 255).astype(np.uint8)
#             else:
#                 aug_mask_image = aug_mask_image.astype(np.uint8)

#             aug_mask_image_path = os.path.join(save_dir, f"{idx}_aug_mask.jpg")
#             cv2.imwrite(aug_mask_image_path, aug_mask_image)

# def get_dataset(preprocessed_input_dir, aug_pipeline=None, data_type='train'):
#     data_dir = os.path.join(preprocessed_input_dir, data_type, 'data')
#     label_dir = os.path.join(preprocessed_input_dir, data_type, 'mask')
#     return LungTumorDataset(data_dir, label_dir, transform=aug_pipeline)






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