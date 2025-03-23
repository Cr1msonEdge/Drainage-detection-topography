from segmentation_models_pytorch import Unet, UnetPlusPlus
from torch.cuda import is_available
from os import listdir
from os.path import isfile, join
from .basemodel import MODELS_PATH, get_counter, BaseModel
from torch import argmax, no_grad, unsqueeze, sigmoid, float
from helper.callbacks.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import torch.nn.functional as F
from helper.models.config import *


class UNet(BaseModel):
    def __init__(self, config: Config, type=None):
        if type in ['Unet', 'UnetImageNet', 'Unet++']:
            super().__init__(type, config)
        else:
            print('Such type of UNet do not exist')
            return
        
        # Getting model
        if type == 'Unet':    
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights=None,  # Randomly initialized weights
                in_channels=3,
                classes=2
            )
        elif type == 'UnetImageNet':
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=2
            )
        elif type == 'Unet++':
            self.model = UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=3,
                classes=2
            )
        
        self.model = self.model.to(self.device)
                
        # Получение индекса
        self.counter = get_counter(self.model_name)
    
    def compute_outputs(self, image):
        return super().compute_outputs(image)
    