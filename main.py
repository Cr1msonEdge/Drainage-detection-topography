from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=10, dataset_name='test_bad')
    name = run_training("Segformer", config)
    run_test(f"{name}.ckpt")
