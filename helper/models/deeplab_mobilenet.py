from .basemodel import BaseModel
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from segmentation_models_pytorch import DeepLabV3Plus
import torch
from helper.models.config import *

        
class DeepLab(BaseModel):
    def __init__(self, config: Config=None):
        super().__init__("DeepLabV3", config)
        
        self.model = DeepLabV3Plus(encoder_weights="imagenet", in_channels=4, classes=2)

        # Adapting to channels
        self.model.encoder.conv1 = self.adapt_conv_layer(self.model.encoder.conv1, in_channels=config.num_channels)
        
        # Freezing layers except of first conv layer
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.encoder.conv1.parameters():
            param.requires_grad = True
        self.model = self.model.to(self.base_device)

    def compute_outputs(self, images):
        return self.model(images)["out"]

    def forward(self, images):
        return self.model(images)["out"]