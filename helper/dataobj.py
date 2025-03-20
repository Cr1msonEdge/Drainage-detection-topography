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
        self.transform = None
        if device is None:
            self.device = 'cuda' if is_available() else 'cpu'
        else:
            self.device = device
        self.mode = mode
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.mode == 'train':    
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.25),
                A.RandomRotate90(p=0.25),
            ])
        else:
            self.transform = A.Compose([])
            
        data = self.transform(image=image, mask=mask)
        
        if self.mode == 'train':
            color_jitter = A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), dtype=np.uint8)
            image = color_jitter(image=data['image'])['image']
        
        mask = data['mask']
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        mask = np.transpose(mask, (2,0,1)).astype(np.float32)
        
        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0).long()
        
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
            
            plt.subplot(1,2,1)
            plt.imshow(np.transpose(image, (1,2,0)))
            plt.axis('off')
            plt.title("Original Image")

            plt.subplot(1,2,2)
            plt.imshow(np.transpose(mask, (1,2,0)), cmap='gray')
            plt.axis('off')
            plt.title("True Mask")
            
        else:
            # Numpy arrays
            image, mask = self.images[idx], self.masks[idx]
            
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title("Original Image")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title("True Mask")
        
        plt.show()
        