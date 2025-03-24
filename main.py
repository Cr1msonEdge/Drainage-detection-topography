from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=1)
    run_training("Unet", config)
