from torch import save, load, unsqueeze, argmax, no_grad, stack
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import pathlib
from helper.callbacks.visualize import show_prediction
from helper.callbacks.metrics import get_iou
from helper.models.config import Config
from os import listdir
from os.path import isfile, join
from helper.callbacks.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import lightning as pl
import abc
from helper.models.config import Config
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import uuid


MODELS_PATH = str(pathlib.Path(__file__).parent.resolve()) + '/saved'


class BaseModel(pl.LightningModule):
    def __init__(self, model_name, config=Config(), verbose=0):
        self.model_name = model_name
        print(f"Initialized {model_name}")
        
        self.config = config
        if self.config.loss == 'CrossEntropy':
            self.loss = CrossEntropyLoss    
        print(f"{model_name} initialized with hyperparams: {config.get_params()}")
        
        # Saving hparamas to log
        hparams = {
            'name': model_name, 
            **self.config.to_dict()
        }
        self.save_hyperparameters(**hparams)
        
        # Saving output during epoch
        
        self.training_step_loss = []
        self.training_step_metric = []
        self.validation_step_loss = []
        self.validation_step_metric = []
        
        self.unique_id = str(uuid.uuid4())[:8]

    def set_id(self, id):
        self.unique_id = id
    
    @abc.abstractmethod
    def forward(self, images):
        pass
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        loss = self.loss(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_loss.append(loss)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True)

        preds = argmax(outputs, dim=1)
        metric = get_iou(preds, masks)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True)   
        self.training_step_metric.append(metric)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        loss = self.loss(outputs, masks)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_loss.append(loss)

        preds = argmax(outputs, dim=1)
        metric = get_iou(preds, masks)
        self.log('val_metric', metric, on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_metric.append(metric)
        
        return {
            'val_loss': loss,
            'val_metric': metric
        }        
        
    
    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_step_loss) / len(self.train_step_loss)
        avg_train_metric = sum(self.train_step_metric) / len(self.train_step_metric)
        
        self.log('avg_train_loss', avg_train_loss)
        self.log('avg_train_metric', avg_train_metric)

        self.training_step_loss.clear()
        self.training_step_metric.clear()
    
    def on_validation_epoch_end(self):
        avg_validation_loss = sum(self.validation_step_loss) / len(self.validation_step_loss)
        avg_validation_metric = sum(self.validation_step_metric) / len(self.validation_step_metric)
        
        self.log('avg_val_loss', avg_validation_loss)
        self.log('avg_val_metric', avg_validation_metric)

        self.validation_step_loss.clear()
        self.validation_step_metric.clear()
    
    
    def configure_optimizers(self):
        if self.config.optimizer == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.optimizer == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        elif self.config.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        if self.config.scheduler is not None:
            if self.config.scheduler == 'Plateau':    
                return (
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            ReduceLROnPlateau(optimizer, **self.config.scheduler_params)
                        }
                    }
                )
            
            if self.config.scheduler == 'MultistepLR':
                return (
                    {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            MultiStepLR(optimizer, **self.config.scheduler_params)
                        }
                    }
                )    
        
        return optimizer
        
            
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
    