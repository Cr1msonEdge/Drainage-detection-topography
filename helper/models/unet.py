from segmentation_models_pytorch import Unet, UnetPlusPlus
from .basemodel import BaseModel
from helper.models.config import *
import torch


class UNet(BaseModel):
    def __init__(self, config: Config=None, type=None):
        if type is None:
            type = 'Unet'
        assert type in ['Unet', 'UnetImageNet', 'Unet++'], f"Such type of UNet does not exist: {type}"
        super().__init__(type, config)  
        
        # Getting model
        if type == 'Unet':    
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights=None,  
                classes=2
            )
        elif type == 'UnetImageNet':
            self.model = Unet(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                classes=2
            )
        elif type == 'Unet++':
            self.model = UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights=None,
                classes=2
            )
        
        # Adapting to 3 or 4 channels
        self.model.encoder.conv1 = self.adapt_conv_layer(self.model.encoder.conv1, in_channels=config.num_channels)
        
        # Freezing layers except of first conv
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.encoder.conv1.parameters():
            param.requires_grad = True
        
        self.model = self.model.to(self.base_device)


    def forward(self, images):
        return self.model(images)

    def compute_outputs(self, image):
        return super().compute_outputs(image)
    