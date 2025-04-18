from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=50, dataset_name='test_bad', num_channels=4)
    name = run_training("Segformer", config)
    run_test(f"{name}.ckpt")
