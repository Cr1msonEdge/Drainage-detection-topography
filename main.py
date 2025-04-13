from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=1, dataset_name='test_bad')
    run_training("Unet", config)
    # run_test("Unet-20250326-130230.ckpt")
