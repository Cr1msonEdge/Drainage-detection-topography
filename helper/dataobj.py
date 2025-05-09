from torch.utils.data import Dataset
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import is_available
import albumentations as A


class DrainageDataset(Dataset):
    """
    Class to work with images 
    """
    def __init__(self, images, masks, device=None, mode='train'):
        self.images = np.array(images)
        self.masks = np.array(masks) 
        # Transforming
        self.images = self.images.astype(np.float32) / 255.0
        self.masks = (self.masks > 0).astype(np.long)
        
        self.transform = None
        if device is None:
            self.device = 'cuda' if is_available() else 'cpu'
        else:
            self.device = device
        
        self.mode = mode
        self.num_channels = self.images.shape[-1]  # For 3 or 4 channel images
        self.color_jitter = A.ColorJitter(brightness=0.2, contrast=0.2) 

        # Normalization 
        # self.rgb_mean = [0.624, 0.642, 0.624]
        # self.rgb_std = [0.2, 0.2, 0.2]
        # self.dem_mean = [0.432]
        # self.dem_std = [0.087]
        
        # if self.num_channels == 4:
        #     self.channels_means = self.rgb_mean + self.dem_mean
        #     self.channels_stds = self.rgb_std + self.dem_std
        # else:
        #     self.channels_means = self.rgb_mean
        #     self.channels_stds = self.rgb_std
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]  # shape: (H, W, 4)
        mask = self.masks[idx]  # shape: (H, W)

        rgb = image[:, :, :3]
        if self.num_channels == 4:
            dem = image[:, :, 3:]


        if self.mode == 'train':
            rgb_jittered = self.color_jitter(image=rgb)['image']
            
            if self.num_channels == 4:
                image_aug = np.concatenate([rgb_jittered, dem], axis=2)
            else:
                image_aug = rgb_jittered
                
            transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.25),
                    A.RandomRotate90(p=0.25),
                ])
            
        else:
            if self.num_channels == 4:
                image_aug = np.concatenate([rgb, dem], axis=2)
            else:
                image_aug = rgb

            transform = A.Compose([])
            
        data = transform(image=image_aug, mask=mask)
        image_aug = data['image']
        mask = data['mask']

        image_aug = np.transpose(image_aug, (2, 0, 1))
        image = torch.tensor(image_aug, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

    def get_images(self):
        return self.images
    
    def get_masks(self):
        return self.masks
    
    def show(self, idx, augmented=False):
        """
        Show image and mask
        
        Params:
        idx - index of the image from dataset
        processed - if True - processed shows augmented image. If False - shows the original
        """
        if augmented:
            image, mask = self[idx]            
            image = image.numpy()
            mask = mask.numpy()
            print(image.shape)
            print(mask.shape)
            rgb = np.transpose(image[:3, :, :], (1, 2, 0))
            print(rgb.shape)
            dem = image[3, :, :] if self.num_channels == 4 else None
            print(f"dem = {dem}")
        else:
            image = self.images[idx]  # (H, W, 4)
            mask = self.masks[idx]   # (H, W)
            rgb = image[:, :, :3]
            dem = image[:, :, 3] if self.num_channels == 4 else None
        
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3 if dem is not None else 2, 1)
        plt.imshow(rgb)
        plt.axis('off')
        plt.title("RGB Image")

        if dem is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(dem, cmap='terrain')
            plt.axis('off')
            plt.title("DEM")

            plt.subplot(1, 3, 3)
        else:
            plt.subplot(1, 2, 2)

        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title("True Mask")

        plt.tight_layout()
        plt.show()
        