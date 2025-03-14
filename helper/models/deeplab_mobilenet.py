from torch.cuda import is_available
from os import listdir
from os.path import isfile, join
from .basemodel import MODELS_PATH, BaseModel, get_counter
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch import argmax, unsqueeze, no_grad
from torch.nn import Conv2d
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from helper.trainval.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
import torch.nn.functional as F
from helper.callbacks.visualize import show_prediction

        
class DeepLab(BaseModel):
    def __init__(self, device=None):
        super().__init__("DeepLabV3")
        if device is None:
            self.device = 'cuda' if is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = Conv2d(256, 2, kernel_size=1)
        self.model = self.model.to(self.device)
                
        # Получение индекса
        self.counter = get_counter(self.model_name)
    
    def compute_outputs(self, images):
        return self.model(images)["out"]
    
    # def train_epoch(self, dataloader, criterion, optimizer, device):
    #     self.model.train()
        
    #     running_loss = 0.0
    #     running_corrects = 0.0
    #     total_batches = 0
        
    #     for images, masks in dataloader:
    #         images = images.to(device)
    #         masks = masks.to(device).squeeze()
            
    #         outputs = self.compute_outputs(images)
            
    #         loss = criterion(outputs, masks)

    #         outputs = argmax(outputs, dim=1)        
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        
    #         running_loss += loss.item() * images.size(0)    
    #         running_corrects += get_iou(outputs, masks)
    #         total_batches += 1

    #     epoch_loss = running_loss / len(dataloader.dataset)
    #     epoch_acc = running_corrects / total_batches
        
    #     return epoch_loss, epoch_acc.cpu().item()
    
    
    # def val_epoch(self, dataloader, criterion, optimizer, device):
    #     self.model.eval()
    #     val_loss = 0.0
    #     val_corrects = 0.0
    #     total_batches = 0
        
    #     with no_grad():
    #         for images, masks in dataloader:
    #             images = images.to(device)
    #             masks = masks.to(device).squeeze()
                
    #             outputs = self.model(images)['out']
    #             loss = criterion(outputs, masks)
    #             val_loss += loss.item() * images.size(0)
                
    #             predicted = argmax(outputs, dim=1)
    #             val_corrects += get_iou(predicted, masks)
                
    #             total_batches += 1
        
    #     val_loss /= len(dataloader.dataset)
    #     val_acc = val_corrects / total_batches
        
    #     return val_loss, val_acc.cpu().item()
        

    # def train(self, dataloaders, config, device):
    #     print(f"Training model {self.get_name()} - {self.counter} using {device}")
    #     history = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'train_acc': [],
    #         'val_acc': []
    #     }
        
    #     plt.ioff()
        
    #     fig, ax = plt.subplots(1, 1)
    #     hdisplay = display.display('', display_id=True)
    #     ax.set_xlabel('Epoch')
    #     ax.set_ylabel('Loss')
    #     ax.set_title('Training and Validation Loss')
        
    #     train_loss_line, = ax.plot([], [], label='Train Loss')
    #     val_loss_line, = ax.plot([], [], label='Validation Loss')
    #     ax.legend()
        
    #     with tqdm(desc="epoch", total=config.NUM_EPOCHS) as pbar_outer:
    #         optimizer = config.optimizer
    #         criterion = config.criterion
            
    #         for epoch in range(config.NUM_EPOCHS):
    #             train_loss, train_acc = self.train_epoch(dataloaders['train'], criterion, optimizer, device)
                
    #             val_loss, val_acc = self.val_epoch(dataloaders['validate'], criterion, optimizer, device)
                
    #             history['train_loss'].append(train_loss)
    #             history['train_acc'].append(train_acc)
    #             history['val_loss'].append(val_loss)
    #             history['val_acc'].append(val_acc)
                
    #             pbar_outer.update(1)
    #             print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}")
                
    #             train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
    #             val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
                
    #             ax.set_xlim(0, config.NUM_EPOCHS + 2)
    #             ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
                
    #             fig.canvas.draw()
    #             fig.canvas.flush_events()
    #             hdisplay.update(fig)
                
    #     return history


    # def predict(self, image, mask, device, show_full=False, show=False):
    #     assert not(show_full and not(show)), "Show can't be true, while show_full is mode is active"
    #     self.model.eval()  
        
    #     # Converting the images into tensors and send them to the desired device.
    #     image = image.to(device)
    #     mask = mask.to(device)
    #     image = unsqueeze(image, 0)
    #     mask = unsqueeze(mask, 0)
    #     with no_grad():        
            
    #         output = self.compute_outputs(image)
    #         pred = argmax(output, dim=1).squeeze(0).cpu()  # Get the prediction
    #         if show_full:
    #             show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=True)
    #         elif show:
    #             show_prediction(image.squeeze().cpu(), pred, mask.squeeze().cpu(), show_intersection=False)

    #     return pred


    # def test_epoch(self, dataloader, criterion, device, detailed=False):
    #     self.model.to(device)
    #     test_loss = 0.0
    #     total_batches = 0
        
    #     test_iou = 0.0
    #     test_acc = 0.0
    #     test_prec = 0.0
    #     test_recall = 0.0
    #     test_f1 = 0.0
    #     test_dice = 0.0
        
    #     with no_grad():
    #         for images, masks in dataloader:
    #             images = images.to(device)
    #             masks = masks.to(device).squeeze()
                
    #             outputs = self.model(images)['out']
    #             loss = criterion(outputs, masks)
    #             test_loss += loss.item() * images.size(0)
                
    #             predicted = argmax(outputs, dim=1)
    #             test_iou += get_iou(predicted, masks)
    #             if detailed:
    #                 test_acc += get_acc(predicted, masks)
    #                 test_prec += get_prec(predicted, masks)
    #                 test_recall += get_recall(predicted, masks)
    #                 test_f1 += get_f1(predicted, masks)
    #                 test_dice += get_dice(predicted, masks)
                
    #             total_batches += 1
        
    #     test_loss /= len(dataloader.dataset)
    #     test_iou = (test_iou / total_batches)
        
    #     if detailed:
    #         test_acc = test_acc / total_batches
    #         test_prec = test_prec / total_batches
    #         test_recall = test_recall / total_batches
    #         test_f1 = test_f1 / total_batches
    #         test_dice = test_dice / total_batches

    #         return  {'loss': test_loss, 'acc': test_acc.cpu().numpy().item(), 'prec': test_prec.cpu().numpy().item(), 'recall': test_recall.cpu().numpy().item(), 'f1': test_f1.cpu().numpy().item(), 'dice': test_dice.cpu().numpy().item(), 'iou': test_iou.cpu().numpy().item()}
        
    #     return {'loss': test_loss, 'iou': test_iou.cpu()}
        