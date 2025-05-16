from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from .basemodel import BaseModel
import torch
import torch.nn.functional as F
from torch import nn
from helper.models.config import Config
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class NvidiaSegformer(BaseModel):
    def __init__(self, config: Config=None):
        super().__init__('Segformer', config)

        self.id2label = {0: 'background', 1: 'drainage'}
        self.label2id = {label: id for id, label in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True, id2label=self.id2label, label2id=self.label2id)
        
        self.model.segformer.encoder.patch_embeddings[0].proj = self.adapt_conv_layer(self.model.segformer.encoder.patch_embeddings[0].proj, in_channels=self.config.num_channels)
        
        # Freezing encoder except of first layer
        # for name, param in self.model.segformer.encoder.named_parameters():
        #     if "patch_embeddings.0.projection" not in name:
        #         param.requires_grad = False

        self.init_training_components()
        self.model.to(self.device)
        
        #self.feature_extractor = SegformerFeatureExtractor(do_resize=True, size=(256, 256))

    def compute_outputs(self, images):
        outputs = self.model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        return outputs
