import argparse
import logging
from helper.models.config import Config
from engine.model_utils import *
from helper.models.unet import *
from helper.models.deeplab_mobilenet import *
from helper.models.nvidia_ade20k import *
from engine.data_setup import *
from engine.model_utils import *
import mlflow
import mlflow.pytorch


def run_training(model_name, config: Config, tags=None, description=None):
    """
    Run training
    Return: name of the model
    """

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
    dataloader = get_dataloader(mode='train', device=config.device, batch_size=config.batch_size, name=config.dataset_name, channels=config.num_channels, clahe=False, dilate_mask=False)
    val_loader = get_dataloader(mode='val', device=config.device, batch_size=config.batch_size, name=config.dataset_name, channels=config.num_channels, clahe=False, dilate_mask=False)

    print("=== Setting up MLflow ===")
    mlflow.set_experiment(config.dataset_name)
    with mlflow.start_run(run_name=f"{model_name}-{model.unique_id}") as run:
        mlflow.log_params({
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "dataset": config.dataset_name,
            "lr": config.learning_rate,
            "scheduler": config.scheduler,
            "optimizer": config.optimizer,
            "model": model_name,
        })

        if tags:
            mlflow.set_tags(tags)
        if description:
            mlflow.set_tag("description", description)

        model.logger = mlflow

        print("=== Training ===")
        # Training
        model.train_loop(dataloader, val_loader)
        print("=== Training finished ===")
        print(f"Saved into {get_model_folder(model_name, verbose=-1)}/{model_name}-{model.unique_id}.pt")
        print(f"Config is {model.config}")

        mlflow.pytorch.log_model(model.model, artifact_path="models")
        
    return f"{model_name}-{model.unique_id}"

if __name__ == "__main__":
    print('asd')
