import argparse
import logging
from helper.models.config import Config
from engine.model_utils import *
from helper.callbacks.metrics import NeptuneCallback
from helper.models.unet import *
from helper.models.deeplab_mobilenet import *
from helper.models.nvidia_ade20k import *
from engine.data_setup import *
from engine.model_utils import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from helper.api_key import *


class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )




def run_training(model_name, config: Config, tags=None, description=None):
    """
    Run training
    Return: name of the model
    """
    
    logging.getLogger("neptune").addFilter(_FilterCallback())
    
    print("=== Starting training ===")
    print("=== Creating model === ")
    # Setting up model
    assert model_name in MODEL_NAMES, f"Model {model_name} is not found."
    if "Unet" in model_name:
        model = UNet(config, type=model_name)
    elif model_name == "Segformer":
        model = NvidiaSegformer(config)
    else:
        model = DeepLab(config)
    
    print("=== Setting up dataloader === ")
    # Getting dataloader
    dataloader = get_dataloader(mode='train', device=config.device, batch_size=config.BATCH_SIZE, name=config.dataset_name)
    val_loader = get_dataloader(mode='val', device=config.device, batch_size=config.BATCH_SIZE, name=config.dataset_name)
    
    print("=== Setting up Neptune === ")
    # Setting up Neptune
    neptune_logger = NeptuneLogger(
        project=PROJECT_NAME,
        api_key=API_TOKEN,
        log_model_checkpoints=False
    )
    
    print("=== Creating checkpoint ===")
    # Saving model
    checkpoint_callback = ModelCheckpoint(
        dirpath=get_model_folder(model_name, verbose=0),
        filename=f'{model_name}-{model.unique_id}',
        save_top_k=1,
        monitor='val_loss'
    )


    # Adding additional information
    neptune_logger.experiment["sys/tags"].add("segmentation")
    neptune_logger.experiment["sys/tags"].add(model_name)
    neptune_logger.experiment["sys/tags"].add(config.dataset_name)
    info = {
        "notes": "Fixing bugs"
    }
    if description is not None:
        info["description"] =  description
    if tags is not None:
        info.update(tags)

    neptune_logger.experiment["extra/info"] = info

    print("=== Training ===")
    # Training
    trainer = Trainer(
        logger=neptune_logger, 
        max_epochs=config.NUM_EPOCHS, 
        callbacks=[checkpoint_callback], 
        devices=1, 
        accelerator="gpu", log_every_n_steps=len(dataloader),    
    )
    trainer.fit(model, dataloader, val_dataloaders=val_loader)

    print("=== Training finished ===")
    print(f"Saved into {get_model_folder(model_name, verbose=-1)}/{model_name}-{model.unique_id}.ckpt")
    print(f"Config is {model.config}")

    return f"{model_name}-{model.unique_id}"

if __name__ == "__main__":
    print('asd')
