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

    def predict(self, image, mask, device, show_full=False, show=False):
        """
        Return model's predict.

        Args:
            image: source image
            mask: source mask
            device: the device where the model is situated
            show_full: if True shows original mask, prediction and intersection
            show: if True plots the prediction
        """
        assert not(show_full and not show), "Show can't be true, while show_full is mode is active"
        self.model.eval()

        # Converting the images into tensors and send them to the desired device.
        image = image.to(device)
        mask = mask.to(device)
        image = unsqueeze(image, 0)
        mask = unsqueeze(mask, 0)
        with no_grad():
            output = self.compute_outputs(image)
            pred = argmax(output, dim=1).squeeze(0).cpu()  # Get the prediction
            if show_full:
                show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=True)
            elif show:
                show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=False)

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

            
#     def save(self):
#         save(self.model.state_dict(), f"{MODELS_PATH}\\{self.model_name}\\{self.model_name}-{self.counter}.pt")
    
#     def load(self, path=None):
#         if path is None:
#             print('No file name to load.')
#             return
#         assert self.model_name in path, 'Wrong path'
#         self.counter = int(path.rstrip('.pt')[path.find('-') + 1:])
#         state_dict = load(path)
#         self.model.load_state_dict(state_dict=state_dict)    
    
#     def get_name(self):
#         return self.model_name
    
#     def get_json_name(self):
#         return f"{self.model_name}-{self.counter}.json"
    
#     def set_counter(self, counter):
#         self.counter = counter
    
#     def compute_outputs(self, image):
#         return self.model(image)
    
#     def train_epoch(self, dataloader, config: Config, device: str):
#         """
#         Training epoch for a model
        
#         Params
#         :dataloader: - dataloader of images and masks
#         :config: - config that contains optimizer, criterion, learning rate, scheduler etc.
#         :device: - 'cuda' or 'cpu' - 
#         """
#         self.model.train()
        
#         running_loss = 0.0
#         running_corrects = 0.0
#         total_batches = 0
        
#         for images, masks in dataloader:
#             images = images.to(device)
#             masks = masks.to(device).squeeze()
            
#             outputs = self.compute_outputs(images)
            
#             loss = config.criterion(outputs, masks)

#             outputs = argmax(outputs, dim=1)        
#             config.optimizer.zero_grad()
#             loss.backward()
#             config.optimizer.step()
        
#             running_loss += loss.item() * images.size(0)
#             running_corrects += get_iou(outputs, masks)
#             total_batches += 1


#         epoch_loss = running_loss / len(dataloader.dataset)
#         epoch_acc = running_corrects / total_batches
        
#         return epoch_loss, epoch_acc.cpu().item()
    
#     def val_epoch(self, dataloader, config: Config, device):
#         """
#         Validating epoch for a model
        
#         Params
#         :dataloader: - dataloader of images and masks
#         :config: - config that contains optimizer, criterion, learning rate, scheduler etc.
#         :device: - 'cuda' or 'cpu' - 
#         """
#         self.model.eval()
#         val_loss = 0.0
#         val_corrects = 0.0
#         total_batches = 0
        
#         with no_grad():
#             for images, masks in dataloader:
#                 images = images.to(device)
#                 masks = masks.to(device).squeeze()
                
#                 outputs = self.compute_outputs(images)
#                 loss = config.criterion(outputs, masks)
#                 val_loss += loss.item() * images.size(0)
                
#                 predicted = argmax(outputs, dim=1)
#                 val_corrects += get_iou(predicted, masks)
                
#                 total_batches += 1
        
#         val_loss /= len(dataloader.dataset)
#         val_acc = val_corrects / total_batches
        
#         return val_loss, val_acc.cpu().item()
    
#     def test_epoch(self, dataloader, config: Config, device, detailed=False):
#         """
#         Testing epoch for a model
        
#         Params
#         :dataloader: - dataloader of images and masks
#         :config: - config that contains optimizer, criterion, learning rate, scheduler etc.
#         :device: - 'cuda' or 'cpu' 
#         """
#         self.model.to(device)
#         test_loss = 0.0
#         total_batches = 0
        
#         test_iou = 0.0
#         test_acc = 0.0
#         test_prec = 0.0
#         test_recall = 0.0
#         test_f1 = 0.0
#         test_dice = 0.0
        
#         with no_grad():
#             for images, masks in dataloader:
#                 images = images.to(device)
#                 masks = masks.to(device).squeeze()
                
#                 outputs = self.compute_outputs(images)
#                 loss = config.criterion(outputs, masks)
#                 test_loss += loss.item() * images.size(0)
                
#                 predicted = argmax(outputs, dim=1)
#                 test_iou += get_iou(predicted, masks)
#                 if detailed:
#                     test_acc += get_acc(predicted, masks)
#                     test_prec += get_prec(predicted, masks)
#                     test_recall += get_recall(predicted, masks)
#                     test_f1 += get_f1(predicted, masks)
#                     test_dice += get_dice(predicted, masks)

#                 total_batches += 1
        
#         test_loss /= len(dataloader.dataset)
#         test_iou = (test_iou / total_batches)
        
#         if detailed:
#             test_acc = test_acc / total_batches
#             test_prec = test_prec / total_batches
#             test_recall = test_recall / total_batches
#             test_f1 = test_f1 / total_batches
#             test_dice = test_dice / total_batches
            
#             return  {'loss': test_loss, 'acc': test_acc.cpu().numpy().item(), 'prec': test_prec.cpu().numpy().item(), 'recall': test_recall.cpu().numpy().item(), 'f1': test_f1.cpu().numpy().item(), 'dice': test_dice.cpu().numpy().item(), 'iou': test_iou.cpu().numpy().item()}
        
#         return {'loss': test_loss, 'iou': test_iou.cpu()}
    
    
#     def train(self, dataloaders, config, device):
#         """
#         Full training cycle for a model
        
#         Params
#         :dataloaders: - dataloaders consists of training and validating dataloaders
#         :config: - config that contains optimizer, criterion, learning rate, scheduler etc.
#         :device: - 'cuda' or 'cpu' - 
#         """
#         print(f"Training model {self.get_name()} - {self.counter} using {device}")
#         history = {
#             'train_loss': [],
#             'val_loss': [],
#             'train_acc': [],
#             'val_acc': []
#         }
        
#         plt.ioff()
        
#         fig, ax = plt.subplots(1, 1)
#         hdisplay = display.display('', display_id=True)
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Loss')
#         ax.set_title('Training and Validation Loss')
        
#         train_loss_line, = ax.plot([], [], label='Train Loss')
#         val_loss_line, = ax.plot([], [], label='Validation Loss')
#         ax.legend()
        
#         with tqdm(desc="epoch", total=config.NUM_EPOCHS) as pbar_outer:
#             scheduler = None
#             if 'scheduler' in config.get_params().keys():
#                 scheduler = config.scheduler
                
#                 for epoch in range(config.NUM_EPOCHS):
#                     train_loss, train_acc = self.train_epoch(dataloaders['train'], config, device)
                    
#                     val_loss, val_acc = self.val_epoch(dataloaders['validate'], config, device)
                    
#                     history['train_loss'].append(train_loss)
#                     history['train_acc'].append(train_acc)
#                     history['val_loss'].append(val_loss)
#                     history['val_acc'].append(val_acc)
                    
#                     pbar_outer.update(1)
#                     print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}")
#                     scheduler.step(val_loss)
#                     print(f", lr {scheduler.get_last_lr()}")
#                     train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
#                     val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
                    
#                     ax.set_xlim(0, config.NUM_EPOCHS + 2)
#                     ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
                    
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
#                     hdisplay.update(fig)
#             else:
#                 # Without scheduler
#                 for epoch in range(config.NUM_EPOCHS):
#                     train_loss, train_acc = self.train_epoch(dataloaders['train'], config, device)
                    
#                     val_loss, val_acc = self.val_epoch(dataloaders['validate'], config, device)
                    
#                     history['train_loss'].append(train_loss)
#                     history['train_acc'].append(train_acc)
#                     history['val_loss'].append(val_loss)
#                     history['val_acc'].append(val_acc)
                    
#                     pbar_outer.update(1)
#                     print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}")
#                     train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
#                     val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
                    
#                     ax.set_xlim(0, config.NUM_EPOCHS + 2)
#                     ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
                    
#                     fig.canvas.draw()
#                     fig.canvas.flush_events()
#                     hdisplay.update(fig)
#         return history
    
#     def predict(self, image, mask, device, show_full=False, show=False):
#         """
#         Return model's predict. 
        
#         Params:
#             image: source image
#             mask: source mask
#             device: the device where the model is situated
#             show_full: if True shows original mask, prediction and intersection
#             show: if True plots the prediction
#         """
#         assert not(show_full and not(show)), "Show can't be true, while show_full is mode is active"
#         self.model.eval()  
        
#         # Converting the images into tensors and send them to the desired device.
#         image = image.to(device)
#         mask = mask.to(device)
#         image = unsqueeze(image, 0)
#         mask = unsqueeze(mask, 0)
#         with no_grad():        
            
#             output = self.compute_outputs(image)
#             pred = argmax(output, dim=1).squeeze(0).cpu()  # Get the prediction
#             if show_full:
#                 show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=True)
#             elif show:
#                 show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=False)

#         return pred
    
    
# def get_counter(model_name):
#     """
#     Gets the number of model according to existing saved models' files
#     """
#     model_files = [f for f in listdir(MODELS_PATH) if (isfile(join(MODELS_PATH, f)) and join(MODELS_PATH, f).startswith(MODELS_PATH + '\\' + f'{model_name}-'))]
#     files_indexes = [int(f.rstrip('.pt')[f.find('-') + 1:]) for f in model_files]
    
#     if len(files_indexes) != 0:
#         return max(files_indexes) + 1
    
#     return 1
    