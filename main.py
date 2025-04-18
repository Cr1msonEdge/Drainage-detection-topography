from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=100, dataset_name='temp4data', num_channels=4)
    name = run_training("Unet", config)
    run_test(f"{name}.ckpt")
