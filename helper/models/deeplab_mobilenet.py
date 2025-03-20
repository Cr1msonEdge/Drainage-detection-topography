from torch.cuda import is_available
from os import listdir
from os.path import isfile, join
from .basemodel import MODELS_PATH, BaseModel, get_counter
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch import argmax, unsqueeze, no_grad
from torch.nn import Conv2d
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from helper.callbacks.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
import torch.nn.functional as F
from helper.callbacks.visualize import show_prediction

        
class DeepLab(BaseModel):
    def __init__(self, device=None):
        super().__init__("DeepLabV3")
        if device is None:
            self.device = 'cuda' if is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = Conv2d(256, 2, kernel_size=1)
        self.model = self.model.to(self.device)
                
        # Получение индекса
        self.counter = get_counter(self.model_name)
    
    def compute_outputs(self, images):
        return self.model(images)["out"]
        