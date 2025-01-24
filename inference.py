from argparse import ArgumentParser
import os
import cv2
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from models.lit_segmentation_model import LitLungTumorSegModel
from models.segnet import *



# def ct_slices_generator(img_path, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True,
#                         scaling_value=3071):
#     scan_data = nib.load(img_path)
#     ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
#     for idx in range(ct_scan_volume.shape[-1]):
#         if nib.aff2axcodes(scan_data.affine) == orientation:
#             original_shape = ct_scan_volume[:, :, idx].shape
#             resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
#             if vgg_compatible:
#                 resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
#             yield np.moveaxis(resized_data, -1, 0), original_shape
#         else:
#             print(f"{img_path} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")


# def infer(path_to_ckpt, ct_slices, path_to_result_dir, name):
#     model = LitLungTumorSegModel.load_from_checkpoint(path_to_ckpt)
#     # model = torch.load(path_to_ckpt)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval().to(device)
#     results = []
#     for idx, (scan_data, original_shape) in enumerate(ct_slices):
#         with torch.no_grad():
#             mask = model(torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device))
#             resized_mask = cv2.resize(mask.squeeze(0).cpu().numpy(), original_shape, interpolation=cv2.INTER_NEAREST)
#             results.append(resized_mask)
#         np.save(os.path.join(path_to_result_dir, f"{idx}_{name}"), resized_mask)
#     full_mask = np.stack(results, axis=-1)
#     nifti_mask = nib.Nifti1Image(full_mask, affine=np.eye(4))
#     nib.save(nifti_mask, os.path.join(path_to_result_dir, f"label_{name}"))











# def ct_slices_generator(img_path, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True,
#                         scaling_value=3071):
#     scan_data = nib.load(img_path)
#     ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
#     for idx in range(ct_scan_volume.shape[-1]):
#         if nib.aff2axcodes(scan_data.affine) == orientation:
#             original_shape = ct_scan_volume[:, :, idx].shape
#             resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
#             if vgg_compatible:
#                 resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
#             yield np.moveaxis(resized_data, -1, 0), original_shape
#         else:
#             print(f"{img_path} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")


# def infer(path_to_ckpt, ct_slices, path_to_result_dir, name):
#     model = LitLungTumorSegModel.load_from_checkpoint(path_to_ckpt)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.eval().to(device)
    
#     os.makedirs(path_to_result_dir, exist_ok=True)

#     results = []
#     for idx, (scan_data, original_shape) in enumerate(ct_slices):
#         with torch.no_grad():
#             scan_data_tensor = torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device)
#             mask = model(scan_data_tensor).cpu().numpy().squeeze(0)
#             if mask.ndim == 3:
#                 for i in range(mask.shape[0]):
#                     single_slice_mask = mask[i]
#                     resized_mask = cv2.resize(single_slice_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
#                     mask_image = (resized_mask * 255).astype(np.uint8)
#                     mask_image_path = os.path.join(path_to_result_dir, f"{idx}_slice_{i}_mask.jpg")
#                     cv2.imwrite(mask_image_path, mask_image)
            
#             elif mask.ndim == 2:
#                 resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
#                 mask_image = (resized_mask * 255).astype(np.uint8)
#                 mask_image_path = os.path.join(path_to_result_dir, f"{idx}_mask.jpg")
#                 cv2.imwrite(mask_image_path, mask_image)

#             results.append(resized_mask)

#             data_image = np.moveaxis(scan_data, 0, -1)
#             data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image)) * 255
#             data_image = data_image.astype(np.uint8)
#             data_image_path = os.path.join(path_to_result_dir, f"{idx}_data.jpg")
#             cv2.imwrite(data_image_path, data_image)








def ct_slices_generator(img_path, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True,
                        scaling_value=3071):
    scan_data = nib.load(img_path)
    ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
    for idx in range(ct_scan_volume.shape[-1]):
        if nib.aff2axcodes(scan_data.affine) == orientation:
            original_shape = ct_scan_volume[:, :, idx].shape
            resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
            if vgg_compatible:
                resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
            yield np.moveaxis(resized_data, -1, 0), original_shape
        else:
            print(f"{img_path} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")

def overlay_mask_on_image(image, mask):
    """Overlay the mask on the image with a color."""
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)  # Convert mask to binary
    
    # Resize mask to match the image dimensions if necessary
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create an overlay image
    overlay = np.copy(image)
    
    # Define a color for the mask overlay (e.g., red)
    mask_color = np.array([255, 0, 0], dtype=np.uint8)
    
    # Apply the mask
    for c in range(3):  # For each color channel
        overlay[:, :, c] = np.where(mask == 1, mask_color[c], image[:, :, c])
    
    return overlay

def infer(path_to_ckpt, ct_slices, path_to_result_dir, name):
    model = LitLungTumorSegModel.load_from_checkpoint(path_to_ckpt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    
    os.makedirs(path_to_result_dir, exist_ok=True)

    data_images = []
    overlay_images = []

    for idx, (scan_data, original_shape) in enumerate(ct_slices):
        with torch.no_grad():
            scan_data_tensor = torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device)
            mask = model(scan_data_tensor).cpu().numpy().squeeze(0)
            
            # Handle 3D masks if present
            if mask.ndim == 3:
                for i in range(mask.shape[0]):
                    single_slice_mask = mask[i]
                    resized_mask = cv2.resize(single_slice_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Create an overlay of the mask on the original CT slice
                    original_image = np.moveaxis(scan_data, 0, -1)
                    original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image)) * 255
                    original_image = original_image.astype(np.uint8)
                    overlay_image = overlay_mask_on_image(original_image, resized_mask)
                    
                    # Save images
                    mask_image_path = os.path.join(path_to_result_dir, f"{idx}_slice_{i}_mask.jpg")
                    data_image_path = os.path.join(path_to_result_dir, f"{idx}_slice_{i}_data.jpg")
                    overlay_image_path = os.path.join(path_to_result_dir, f"{idx}_slice_{i}_overlay.jpg")
                    
                    cv2.imwrite(mask_image_path, resized_mask * 255)
                    cv2.imwrite(data_image_path, original_image)
                    cv2.imwrite(overlay_image_path, overlay_image)
                    
                    data_images.append(data_image_path)
                    overlay_images.append(overlay_image_path)

            elif mask.ndim == 2:
                resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create an overlay of the mask on the original CT slice
                original_image = np.moveaxis(scan_data, 0, -1)
                original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image)) * 255
                original_image = original_image.astype(np.uint8)
                overlay_image = overlay_mask_on_image(original_image, resized_mask)
                
                # Save images
                mask_image_path = os.path.join(path_to_result_dir, f"{idx}_mask.jpg")
                data_image_path = os.path.join(path_to_result_dir, f"{idx}_data.jpg")
                overlay_image_path = os.path.join(path_to_result_dir, f"{idx}_overlay.jpg")
                
                cv2.imwrite(mask_image_path, resized_mask * 255)
                cv2.imwrite(data_image_path, original_image)
                cv2.imwrite(overlay_image_path, overlay_image)
                
                data_images.append(data_image_path)
                overlay_images.append(overlay_image_path)

    # Create GIFs
    data_gif_path = os.path.join(path_to_result_dir, f"{name}_data.gif")
    overlay_gif_path = os.path.join(path_to_result_dir, f"{name}_overlay.gif")
    
    # Convert images to GIFs
    with Image.open(data_images[0]) as img:
        img.save(data_gif_path, save_all=True, append_images=[Image.open(img) for img in data_images[1:]], duration=30, loop=0)
    
    with Image.open(overlay_images[0]) as img:
        img.save(overlay_gif_path, save_all=True, append_images=[Image.open(img) for img in overlay_images[1:]], duration=30, loop=0)

    print(f"GIFs saved at: {data_gif_path} and {overlay_gif_path}")



name = os.path.basename("/home/mahdi/Desktop/sha/dataset/imagesTs/lung_011.nii.gz")
preprocessed_ct_scan = ct_slices_generator("/home/mahdi/Desktop/sha/dataset/imagesTs/lung_011.nii.gz", (224,224), orientation=('L', 'A', 'S'), vgg_compatible=True)
infer("test.ckpt", preprocessed_ct_scan, "infer", name)










































# from argparse import ArgumentParser
# import os
# import cv2
# import torch
# import numpy as np
# import nibabel as nib
# from PIL import Image
# from models.segnet import SegNet  # Import your model architecture

# def ct_slices_generator(img_path, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True, scaling_value=3071):
#     scan_data = nib.load(img_path)
#     ct_scan_volume = nib.load(img_path).get_fdata() / scaling_value
#     for idx in range(ct_scan_volume.shape[-1]):
#         if nib.aff2axcodes(scan_data.affine) == orientation:
#             original_shape = ct_scan_volume[:, :, idx].shape
#             resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
#             if vgg_compatible:
#                 resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
#             yield np.moveaxis(resized_data, -1, 0), original_shape, ct_scan_volume[:, :, idx]
#         else:
#             print(f"{img_path} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")

# def infer(path_to_model, ct_slices, path_to_result_dir, name):
#     # Load the entire model
#     model = torch.load(path_to_model)
#     model.eval()
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     os.makedirs(path_to_result_dir, exist_ok=True)

#     results = []
#     for idx, (scan_data, original_shape, original_slice) in enumerate(ct_slices):
#         with torch.no_grad():
#             # Add batch dimension and send to device
#             scan_data = torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device)
#             mask = model(scan_data)
#             mask = mask.squeeze(0).cpu().numpy()  # Remove batch dimension
            
#             # Check if mask is 3D
#             if mask.ndim == 3:
#                 for i in range(mask.shape[0]):
#                     slice_mask = mask[i]
                    
#                     # Resize mask to original shape
#                     resized_mask = cv2.resize(slice_mask, original_shape, interpolation=cv2.INTER_NEAREST)
                    
#                     # Ensure the mask is in the right format
#                     resized_mask_uint8 = np.uint8(resized_mask * 255)
                    
#                     # Check if mask is 2D or 3D
#                     if len(resized_mask_uint8.shape) == 2:
#                         mask_image = Image.fromarray(resized_mask_uint8)
#                         mask_image = mask_image.convert('L')  # Convert to grayscale
#                         mask_image.save(os.path.join(path_to_result_dir, f"{idx}_{name}_mask_{i}.jpg"))
#                     else:
#                         print(f"Unexpected mask shape: {resized_mask_uint8.shape}")

#             else:
#                 print(f"Unexpected mask shape: {mask.shape}")

#             # Save original slice image
#             original_slice_uint8 = np.uint8(original_slice / np.max(original_slice) * 255)
#             original_slice_image = Image.fromarray(original_slice_uint8)
#             original_slice_image = original_slice_image.convert('L')  # Convert to grayscale
#             original_slice_image.save(os.path.join(path_to_result_dir, f"{idx}_{name}_slice.jpg"))

#     # Save combined mask image
#     if results:
#         full_mask = np.stack(results, axis=-1)
#         combined_mask_uint8 = np.uint8(full_mask * 255)
#         if len(combined_mask_uint8.shape) == 3 and combined_mask_uint8.shape[-1] == 1:
#             combined_mask_uint8 = combined_mask_uint8.squeeze(-1)  # Convert to 2D if grayscale
        
#         combined_mask_image = Image.fromarray(combined_mask_uint8)
#         combined_mask_image = combined_mask_image.convert('L')  # Convert to grayscale
#         # combined_mask_image.save(os.path.join(path_to_result_dir, f"combined_{name}_mask.jpg"))

# def cli_main():
#     parser = ArgumentParser()
#     parser.add_argument('--path_to_ckpt', type=str)
#     parser.add_argument('--path_to_ct_scan', type=str)
#     parser.add_argument('--path_to_result_dir', type=str, default=os.getcwd())
#     parser.add_argument('--vgg_compatible', type=bool, default=True)
#     parser.add_argument('--resize', type=tuple, default=(224, 224))
#     parser.add_argument('--orientation', type=tuple, default=('L', 'A', 'S'))
#     args = parser.parse_args()

#     name = os.path.basename("/home/mahdi/Desktop/sha/dataset/imagesTs/lung_013.nii.gz")
#     preprocessed_ct_scan = ct_slices_generator("/home/mahdi/Desktop/sha/dataset/imagesTs/lung_013.nii.gz", args.resize, args.orientation, args.vgg_compatible)
#     infer("maze/best_model.pth", preprocessed_ct_scan, "infer", name)

# if __name__ == '__main__':
#     cli_main()
