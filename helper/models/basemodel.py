import numpy as np
import rasterio
from torch import unsqueeze, argmax, no_grad
from torch.nn import Conv2d, init
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
from engine.model_utils import get_model_folder
from helper.callbacks.visualize import show_prediction
from helper.dataobj import *

from helper.callbacks.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import abc
from helper.models.config import Config
import os
import time
from torch import nn


MODELS_PATH = str(pathlib.Path(__file__).parent.resolve()) + '/saved'


class BaseModel:
    def __init__(self, model_name, config=None, logger=None):
        """
        Params:
        model_name - name of the type of model (Segformer, Unet, DeepLab)
        config - configuration of class Config(). If none means the model is loading from checkpoint
        """
        super().__init__()

        self.model_name = model_name
        self.config = config
        self.device = self.config.device
        self.compute_loss = nn.CrossEntropyLoss()

        self.unique_id = config.uid if hasattr(config, "uid") is not None else self.generate_id()

        # Model's parameters
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Logging
        self.logger = logger

    def init_training_components(self):
        assert self.model is not None, "self.model must be defined before calling init_training_components()"
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()
        self.criterion = self.configure_loss()

    def generate_id(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return timestamp

    def adapt_conv_layer(self, conv_layer, in_channels: int=4):
        """
        Adapt the input convolutional layer to number channels
        """
        if conv_layer.in_channels == in_channels:
            return conv_layer

        new_conv = Conv2d(
            in_channels=in_channels,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None)
        )

        with no_grad():
            new_conv.weight[:, :3, :, :] = conv_layer.weight[:, :3, :, :]

            # Остальные инициализируем нормально
            if in_channels > 3:
                if self.model_name == 'Segformer':
                    xavier_normal_(new_conv.weight[:, 3:, :, :])
                else:
                    kaiming_normal_(new_conv.weight[:, 3:, :, :], mode='fan_in', nonlinearity='relu')

                # init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.01)
            if conv_layer.bias is not None:
                new_conv.bias.copy_(conv_layer.bias)

        return new_conv

    # ---------------------------------------
    # Model's parameters

    def configure_optimizer(self):
        if self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def configure_scheduler(self):
        if self.config.scheduler is None:
            return None
        if self.config.scheduler.lower() == "steplr":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        elif self.config.scheduler.lower() == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def configure_loss(self):
        return nn.CrossEntropyLoss()

    # ---------------------------------------
    # Model's functions

    @abc.abstractmethod
    def compute_outputs(self, images):
        pass

    def predict(self, image, mask, device, show=False):
        """
        Return model's predict.

        Args:
            image: source image
            mask: source mask
            device: the device where the model is situated
            show_full: if True shows original mask, prediction and intersection
            show: if True plots the prediction
        """

        self.model.eval()

        # Converting the images into tensors and send them to the desired device.
        image = image.to(device)
        mask = mask.to(device)
        image = unsqueeze(image, 0)
        mask = unsqueeze(mask, 0)
        with no_grad():
            output = self.compute_outputs(image)
            pred = argmax(output, dim=1).squeeze(0).cpu()  # Get the prediction
            show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), num_channels=image.shape[1])

        return pred

    def validate_loop(self, val_dataloader: DataLoader):
        self.model.eval()
        val_loss = 0.0
        val_metric = 0.0

        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(self.device), masks.to(self.device).squeeze()

                outputs = self.compute_outputs(images)
                loss = self.criterion(outputs, masks)

                preds = argmax(outputs, dim=1)
                val_loss += loss.item()
                val_metric += get_iou(preds, masks)

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_metric = val_metric / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f} | Val Metric: {avg_val_metric:.4f}")

        return avg_val_loss, avg_val_metric


    def train_loop(self, train_loader: DataLoader, val_dataloader: DataLoader = None):
        self.model.train()

        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            running_metric = 0.0

            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device).squeeze()
                self.optimizer.zero_grad()
                outputs = self.compute_outputs(images)
                loss = self.compute_loss(outputs, masks)
                loss.backward()
                self.optimizer.step()
                preds = argmax(outputs, dim=1)
                running_loss += loss.item()
                running_metric += get_iou(preds, masks)

            avg_train_loss = running_loss / len(train_loader)
            avg_train_metric = running_metric / len(train_loader)

            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Metric: {avg_train_metric:.4f}")

            # Validation loop
            val_loss, val_metric = self.validate_loop(val_dataloader)
            if self.scheduler:
                self.scheduler.step()

            self.log_metrics({
                "train_loss": avg_train_loss,
                "train_metric": avg_train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "lr": self.optimizer.param_groups[0]["lr"]
            }, step=epoch)

            print("===============")
        self.save_model()

    def test_loop(self, dataloader, save_chart=True):
        self.model.eval()

        test_loss = 0.0
        total_batches = 0

        test_iou = 0.0
        test_acc = 0.0
        test_prec = 0.0
        test_recall = 0.0
        test_f1 = 0.0
        test_dice = 0.0

        device = self.config.device
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Testing"):
                inputs, masks = images.to(device), masks.to(device).squeeze()

                outputs = self.compute_outputs(inputs)
                loss = self.compute_loss(outputs, masks)

                preds = torch.argmax(outputs, dim=1)

                test_loss += loss
                test_iou += get_iou(preds, masks)
                test_acc += get_acc(preds, masks)
                test_prec += get_prec(preds, masks)
                test_recall += get_recall(preds, masks)
                test_f1 += get_f1(preds, masks)
                test_dice += get_dice(preds, masks)

                total_batches += 1

            metrics = {
                'loss': test_loss / total_batches,
                'acc': test_acc / total_batches,
                'prec': test_prec / total_batches,
                'recall': test_recall / total_batches,
                'f1': test_f1 / total_batches,
                'iou': test_iou / total_batches,
                'dice': test_dice / total_batches
            }


        print("=== Test metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
            if self.logger:
                self.logger.log_metric(f"test_{key}", value)
            metrics[key] = metrics[key].detach().cpu().item()

        if save_chart:
            self.save_test_metrics_bar_chart(metrics, f"{self.model_name}-{self.unique_id}-{time.strftime('%Y%m%d-%H%M%S')}")

        return metrics

    # ----------------------------------------
    # Logging functions
    def set_logger(self, logger):
        self.logger = logger

    def log_metrics(self, metrics: dict, step: int = 0):
        if self.logger:
            for k, v in metrics.items():
                self.logger.log_metric(k, v, step=step)


    def save_model(self):
        model_folder = get_model_folder(self.model_name)
        os.makedirs(os.path.dirname(model_folder), exist_ok=True)
        filename = f"{self.model_name}-{self.unique_id}.pt"
        path = os.path.join(model_folder, filename)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config.to_dict()
        }, path)

        print(f"Model saved to: {path}")

    @classmethod
    def load_model(cls, filename):
        model_folder = get_model_folder(filename.split('-')[0])
        path = os.path.join(model_folder, f"{filename}")
        # print(f"Looking at {str(path)}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu")

        config = Config(**checkpoint["config"])

        model = cls(config)
        model.unique_id = filename.split('-')[1]

        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.model.to(config.device)

        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(config.device)

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            if model.scheduler:
                model.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return model


    # ---------------------------------------------
    # Extra functionality

    def save_test_metrics_bar_chart(self, test_metrics, save_path):
        """
        Save bar chart as the result of
        test_metrics: dict, where key is - name of metric, value - float value.
        save_path: path to save image
        """
        # Get images test folder
        curr_path = Path(__file__).resolve()
        folder = curr_path.parent.parent.parent / 'test_images'

        metric_names = list(test_metrics.keys())
        values = [test_metrics[name] for name in metric_names]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(metric_names, values, color='skyblue')
        plt.xlabel('Метрики')
        plt.ylabel('Значение')
        plt.title('Результаты тестирования')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')

        plt.tight_layout()

        plt.savefig(folder / save_path)

        if self.logger:
            self.logger.log_figure(plt.gcf(), artifact_file=save_path+".png")

        plt.close()

    def split_image(self, image, patch_size=256, stride=128):
        """
        Return patches from cut image
        """
        patches = []
        h, w = image.shape[:2]
        y_positions = list(range(0, h - patch_size + 1, stride))
        x_positions = list(range(0, w - patch_size + 1, stride))

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                patches.append((x, y, patch))

        if y_positions[-1] + patch_size < h:
            y_positions.append(h - patch_size)
        if x_positions[-1] + patch_size < w:
            x_positions.append(w - patch_size)

        return patches

    def merge_patches(self, patches: list, image_shape: tuple, patch_size: int = 256, stride: int = 128) -> np.ndarray:
        """
        Merge patches into one image with overlapping
        """
        h, w = image_shape
        full_mask = np.zeros((h, w), dtype=np.float32)
        weight_mask = np.zeros((h, w), dtype=np.float32)

        for x, y, patch_pred in patches:
            ph, pw = patch_pred.shape
            full_mask[y:y + ph, x:x + pw] = np.logical_or(
                full_mask[y:y + ph, x:x + pw],
                patch_pred.astype(bool)
            )

        return full_mask.astype()

    def predict_image(self, image_path, patch_size=256, stride=128, threshold=0.5, device='cuda', out_filename=None, timed=True, return_mask=False):
        """
        Predicts by whole image. Returns
        """
        self.model.eval()

        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            if out_filename is not None:
                profile = src.profile
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=0
                )
                
        image = image.astype(np.float32) / 255.0
        orig_h, orig_w = image.shape[:2]
        patches = self.split_image(image, patch_size=patch_size, stride=stride)

        coords = []
        tensor_patches = []
        for x, y, patch in patches:
            coords.append((x, y))
            tensor = np.transpose(patch, (2, 0, 1))  # C, H, W
            tensor_patches.append(tensor)

        prob_map = torch.zeros((orig_h, orig_w), dtype=torch.float32, device=device)
        batch_size = 64
        num_patches = len(tensor_patches)
        
        if timed:
            start = time.time()
        
        for i in range(0, num_patches, batch_size):
            batch = np.stack(tensor_patches[i:i + batch_size])
            batch_tensor = torch.from_numpy(batch).float().to(device)  # [B, C, H, W]

            with torch.no_grad():
                output = self.compute_outputs(batch_tensor)  # [B, 1, H, W]
                probs = output[:, 1, ...]
                
            for j in range(probs.shape[0]):
                prob_patch = probs[j]
                x, y = coords[i + j]
                h = min(patch_size, orig_h - y)
                w = min(patch_size, orig_w - x)
                prob_patch = prob_patch[:h, :w]

                # update by max
                prob_map[y:y+h, x:x+w] = torch.maximum(
                    prob_map[y:y+h, x:x+w],
                    prob_patch
                )
        
        if timed:
            print(f"Time to predict: {(time.time() - start):.2f} seconds")                
        
        # Merging into tif
        final_mask = (prob_map > threshold).cpu().numpy().astype(np.uint8)
        
        if out_filename:
            with rasterio.open(out_filename, 'w', **profile) as dst:
                dst.write(final_mask.astype(rasterio.uint8), 1)
        
        if return_mask:
            return final_mask
