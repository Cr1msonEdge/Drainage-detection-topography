from helper.models.config import Config
from engine.train import *
from engine.test import *

if __name__ == "__main__":
    config = Config(num_epochs=10, dataset_name='dataset_plot_light', num_channels=3)
    # name = run_training("Unet", config, tags={'dataset': 'dataset_plot_light'})
    run_test(f"Unet-20250419-192629.pt")
