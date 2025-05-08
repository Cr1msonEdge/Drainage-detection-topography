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
                encoder_name='resnet50',
                encoder_weights='imagenet',
                classes=2
            )
        else:
            raise Exception("Unexpected error when creating Unet.")

        # Adapting to 3 or 4 channels
        self.model.encoder.conv1 = self.adapt_conv_layer(self.model.encoder.conv1, in_channels=self.config.num_channels)

        # Freezing layers except of first conv
        # if type != 'Unet':
        #     for param in self.model.encoder.parameters():
        #         param.requires_grad = False
        #     for param in self.model.encoder.conv1.parameters():
        #         param.requires_grad = True

        self.init_training_components()
        self.model = self.model.to(self.device)


    def forward(self, images):
        return self.model(images)

    def compute_outputs(self, image):
        return self.model(image)
    