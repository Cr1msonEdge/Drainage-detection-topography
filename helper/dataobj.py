from torch.utils.data import Dataset
import torch
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
        self.masks = self.masks.astype(np.long)
        
        self.transform = None
        if device is None:
            self.device = 'cuda' if is_available() else 'cpu'
        else:
            self.device = device
        self.mode = mode
        
    def __len__(self):
        return len(self.images)
    
    # def __getitem__(self, idx):
    #     image = self.images[idx]
    #     mask = self.masks[idx]
    #     if self.mode == 'train':    
    #         self.transform = A.Compose([
    #             A.HorizontalFlip(p=0.5),
    #             A.VerticalFlip(p=0.5),
    #             A.RandomRotate90(p=0.25),
    #             A.RandomRotate90(p=0.25),
    #         ])
    #     else:
    #         self.transform = A.Compose([])
            
    #     data = self.transform(image=image, mask=mask)
        
    #     if self.mode == 'train':
    #         color_jitter = A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))
    #         rgb = data['image'][:, :, :3]  # RGB
    #         dem = data['image'][:, :, 3:]  # DEM
            
    #         rgb_aug = color_jitter(image=rgb)['image']

    #         image = np.concatenate([rgb_aug, dem], axis=2)
        
    #     mask = data['mask']
    #     image = np.transpose(image, (2, 0, 1)).astype(np.float32)        
    #     image = torch.Tensor(image) / 255.0
    #     mask = torch.Tensor(mask)
        
    #     return image, mask
    def __getitem__(self, idx):
        # Загрузка изображения и маски в uint8 формате (диапазон [0,255])
        image = self.images[idx]  # shape: (H, W, 4)
        mask = self.masks[idx]    # shape: (H, W)
        
        # image = image.astype(np.float32) / 255.0
        
        if self.mode == 'train':
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                A.RandomRotate90(p=0.25),
            ])
        else:
            transform = A.Compose([])

        data = transform(image=image, mask=mask)
        image_aug = data['image']
        mask = data['mask']

        # В режиме тренировки применяем ColorJitter к RGB-части
        if self.mode == 'train':
            rgb = image_aug[:, :, :3]  # RGB в [0, 1]
            dem = image_aug[:, :, 3:]  # DEM в [0, 1] тоже (если нужно)

            color_jitter = A.ColorJitter(brightness=0.2, contrast=0.2)  # работает с float32 в [0,1]
            rgb_jittered = color_jitter(image=rgb)['image']

            image_aug = np.concatenate([rgb_jittered, dem], axis=2)

        # Переводим в тензоры
        image_aug = np.transpose(image_aug, (2, 0, 1))  # (C, H, W)
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
            
            # plt.subplot(1,2,1)
            # plt.imshow(np.transpose(image, (1,2,0)))
            rgb = np.transpose(image[:3], (1, 2, 0))
            print(rgb)
            dem = image[3]

            # plt.subplot(1,2,2)
            # # ? plt.imshow(np.transpose(mask, (1,2,0)), cmap='gray')
            # plt.imshow(mask, cmap='gray')
            # plt.axis('off')
            # plt.title("True Mask")
            
        else:
            image = self.images[idx]  # (H, W, 4)
            mask = self.masks[idx]   # (H, W)
            rgb = image[:, :, :3]
            dem = image[:, :, 3]
        
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(rgb)  
        plt.axis('off')
        plt.title("RGB Image")

        plt.subplot(1, 3, 2)
        plt.imshow(dem, cmap='terrain')
        plt.axis('off')
        plt.title("DEM")

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title("True Mask")

        plt.tight_layout()
        plt.show()
        