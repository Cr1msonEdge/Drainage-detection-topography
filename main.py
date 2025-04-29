from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    dataset_name = 'ver5_2012'
    num_channels = 3
    config = Config(num_epochs=200, dataset_name=dataset_name, num_channels=num_channels)
    name = run_training("Unet", config, tags={'mode': 'train', 'dataset': dataset_name, 'extra': ['clahe']})
    run_test(f"{name}.pt", tags={'mode': 'test', 'dataset': dataset_name, 'num_channels': num_channels, 'extra': ['clahe']})

