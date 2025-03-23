import argparse
from helper.models.config import Config
from engine.model_utils import *
from helper.models.unet import *
from helper.models.deeplab_mobilenet import *
from helper.models.nvidia_ade20k import *
from engine.data_setup import *
from engine.model_utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from helper.api_key import *


def run_training(model_name, config: Config):
    """
    Run training
    
    """
    # Setting up model
    assert model_name in MODEL_NAMES, f"Model {model_name} is not found."
    if "Unet" in model_name:
        model = UNet(config, type=model_name)
    elif model_name == "Segformer":
        model = NvidiaSegformer(config)
    else:
        model = DeepLab(config)
        
    # Getting dataloader
    dataloader = get_dataloader(mode='train', device=config.device, batch_size=config.BATCH_SIZE)
    
    # Setting up Neptune
    neptune_logger = NeptuneLogger(
        project=PROJECT_NAME,
        api_key=API_TOKEN,
        log_model_checkpoints=True
    )
    
    # Saving model
    checkpoint_callback = ModelCheckpoint(
        dirpath=get_model_folder(model_name, verbose=0),
        filename=f'{model_name}-{model.unique_id}',
        save_top_k=1,
        monitor='val_loss'
    )

    # Training
    trainer = Trainer(logger=neptune_logger, max_epochs=config.NUM_EPOCHS)
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    print('asd')
