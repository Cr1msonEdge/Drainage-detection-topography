from segmentation_models_pytorch import Unet, UnetPlusPlus
from .basemodel import BaseModel
from helper.models.config import *


class UNet(BaseModel):
    def __init__(self, config: Config=None, type=None):
        if type is None:
            type = 'Unet'
        assert type in ['Unet', 'UnetImageNet', 'Unet++'], f"Such type of UNet does not exist: {type}"
        super().__init__(type, config)  # Вызов конструктора родителя
        
        # Getting model
        if type == 'Unet':    
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights=None,  # Randomly initialized weights
                in_channels=4,
                classes=2
            )
        elif type == 'UnetImageNet':
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=4,
                classes=2
            )
        elif type == 'Unet++':
            self.model = UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights=None,
                in_channels=4,
                classes=2
            )
        
        self.model = self.model.to(self.base_device)
                
    def forward(self, images):
        return self.model(images)

    def compute_outputs(self, image):
        return super().compute_outputs(image)
    