from .basemodel import BaseModel
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.nn import Conv2d
from helper.models.config import *

        
class DeepLab(BaseModel):
    def __init__(self, config: Config):
        super().__init__("DeepLabV3")
        
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = Conv2d(256, 2, kernel_size=1)
        self.model = self.model.to(self.base_device)

    
    def compute_outputs(self, images):
        return self.model(images)["out"]
        