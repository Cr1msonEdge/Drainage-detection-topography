from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    dataset_name = 'ver5'
    config = Config(num_epochs=50, dataset_name=dataset_name, num_channels=4)
    name = run_training("Unet", config, tags={'mode': 'train', 'dataset': dataset_name})
    run_test(f"{name}.pt", tags={'mode': 'test', 'dataset': dataset_name})

