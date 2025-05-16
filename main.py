import warnings
warnings.simplefilter("ignore", UserWarning)
from engine.train import *
from engine.test import *

# MODEL_NAMES = [
#     "Unet",
#     "UnetImageNet",
#     "Unet++",
#     "Segformer",
#     "DeepLabV3"
# ]


if __name__ == "__main__":
    # experiment = mlflow.get_experiment_by_name("ee2_con")
    # print(experiment.experiment_id)
    model_name = "Segformer"
    dataset_name = 'ver8'
    num_channels = 4
    config = Config(num_epochs=200, dataset_name=dataset_name, num_channels=num_channels)
    name = run_training(model_name, config, tags={'model': model_name, 'mode': 'train', 'dataset': dataset_name, 'num_channels': num_channels, 'up': 'init'})
    run_test(f"{name}.pt", tags={'model': model_name, 'mode': 'test', 'dataset': dataset_name, 'num_channels': num_channels, 'up': 'init'})
    # run_test(f"Unet-20250509-135948.pt", tags={'model': model_name, 'mode': 'test', 'dataset': dataset_name, 'num_channels': num_channels})

