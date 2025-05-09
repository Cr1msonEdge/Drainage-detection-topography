import argparse
from helper.models.config import Config
from engine.model_utils import *
from helper.models.unet import *
from helper.models.deeplab_mobilenet import *
from helper.models.nvidia_ade20k import *
from engine.data_setup import *
from engine.model_utils import *
import mlflow
from mlflow import start_run, log_params, log_metrics


def run_test(model_name, tags=None):
    # Setting up model
    # assert model_name in MODEL_NAMES, f"Model {model_name} is not found."
    print("=== Starting test ===")

    # Загружаем путь до модели и саму модель
    file_path = get_model_file_path(model_name)

    if "Unet" in model_name:
        model = UNet.load_model(model_name)
    elif "Segformer" in model_name:
        model = NvidiaSegformer.load_model(model_name)
    elif "DeepLab" in model_name:
        model = DeepLab.load_model(model_name)
    else:
        raise Exception(f"Model {model_name} is not recognized.")

    print("=== Model loaded successfully ===")
    print(f"Using config: {model.config}")

    # Загружаем тестовый даталоадер
    if tags and 'dataset' in tags.keys():
        test_loader = get_dataloader(
            mode="test",
            device=model.config.device,
            batch_size=model.config.batch_size,
            name=tags['dataset'],
            channels=model.config.num_channels
        )
    else:
        test_loader = get_dataloader(
            mode="test",
            device=model.config.device,
            batch_size=model.config.batch_size,
            name=model.config.dataset_name,
            channels=model.config.num_channels
        )

    with mlflow.start_run(run_name=f"{model_name}_test"):
        if tags:
            mlflow.set_tags(tags)
        
        print("=== Running test ===")
        model.set_logger(mlflow)
        test_metrics = model.test_loop(test_loader)

        print("=== Test finished ===")

if __name__ == "__main__":
    print('asd')
