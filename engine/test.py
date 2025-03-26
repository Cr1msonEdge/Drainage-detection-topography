import argparse
from helper.models.config import Config
from engine.model_utils import *
from helper.models.unet import *
from helper.models.deeplab_mobilenet import *
from helper.models.nvidia_ade20k import *
from engine.data_setup import *
from engine.model_utils import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from helper.api_key import *


def run_test(model_name, config=None):
    # Setting up model
    # assert model_name in MODEL_NAMES, f"Model {model_name} is not found."
    file_path = get_model_file_path(model_name)
    print(f" Looking in {file_path}")
    if "Unet" in model_name:
        model = UNet.load_from_checkpoint(file_path)
        print("Created Unet")
    elif model_name == "Segformer":
        model = NvidiaSegformer.load_from_checkpoint(file_path)
    else:
        model = DeepLab.load_from_checkpoint(file_path)

    print("=== Loading model ===")
    print(f"Checking {model.hparams}")
    # Getting dataloader
    dataloader = get_dataloader(mode='test', device=model.config.device, batch_size=model.config.BATCH_SIZE)
    
    # Setting up Neptune
    neptune_logger = NeptuneLogger(
        project=PROJECT_NAME,
        api_key=API_TOKEN,
        log_model_checkpoints=True
    )

    # Testing
    trainer = Trainer(logger=neptune_logger, max_epochs=model.config.NUM_EPOCHS)
    trainer.test(model, dataloader)

if __name__ == "__main__":
    print('asd')
