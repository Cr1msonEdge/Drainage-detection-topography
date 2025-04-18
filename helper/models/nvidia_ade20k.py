from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from .basemodel import BaseModel
import torch
import torch.nn.functional as F
from torch import nn

from helper.models.config import Config


class NvidiaSegformer(BaseModel):
    def __init__(self, config: Config=None):
        super().__init__('NvidiaSegformer', config)

        self.id2label = {0: 'background', 1: 'drainage'}
        self.label2id = {label: id for id, label in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True, id2label=self.id2label, label2id=self.label2id)
        
        self.model.segformer.encoder.patch_embeddings[0].proj = self.adapt_conv_layer(self.model.segformer.encoder.patch_embeddings[0].proj, in_channels=self.config.num_channels)
        
        # Freezing encoder except of first layer
        for name, param in self.model.segformer.encoder.named_parameters():
            if "patch_embeddings.0.projection" not in name:
                param.requires_grad = False

        self.init_training_components()
        self.model.to(self.base_device)
        
        self.feature_extractor = SegformerFeatureExtractor(do_resize=True, size=(256, 256))
    
    def forward(self, images):
        outputs = self.model(pixel_values=images).logits
        outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        return outputs

    # def train_epoch(self, dataloader, criterion, optimizer, device):
    #     self.model.train()
        
    #     running_loss = 0.0
    #     running_corrects = 0.0
    #     total_batches = 0
        
    #     for images, masks in dataloader:
    #         images = images.to(device)
    #         masks = masks.to(device).squeeze()
            
    #         outputs = self.compute_outputs(images, masks)
    #         loss = criterion(outputs, masks)

    #         outputs = argmax(outputs, dim=1)        
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
        
    #         running_loss += loss.item() * images.size(0)
    #         running_corrects += get_iou(outputs, masks.squeeze())
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
    #             masks = masks.to(device)
                
    #             outputs = self.model(pixel_values=images).logits
    #             outputs = F.interpolate(outputs, size=(masks.shape[-2:]), mode='bilinear', align_corners=False)
    #             loss = criterion(outputs, masks.squeeze())
    #             val_loss += loss.item() * images.size(0)
    #             #scheduler.step(loss)
    #             predicted = argmax(outputs, dim=1)
    #             #print(f"Val predicted {predicted}, masks val{masks.squeeze()}")
    #             val_corrects += get_iou(predicted, masks.squeeze())
                
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
    #         scheduler = None
    #         if 'scheduler' in config.get_params().keys():
    #             scheduler = config.scheduler
                
    #             for epoch in range(config.NUM_EPOCHS):
    #                 train_loss, train_acc = self.train_epoch(dataloaders['train'], criterion, optimizer, device)
                    
    #                 val_loss, val_acc = self.val_epoch(dataloaders['validate'], criterion, optimizer, device)
                    
    #                 history['train_loss'].append(train_loss)
    #                 history['train_acc'].append(train_acc)
    #                 history['val_loss'].append(val_loss)
    #                 history['val_acc'].append(val_acc)
                    
    #                 pbar_outer.update(1)
    #                 print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}", end='')
    #                 scheduler.step(val_loss)
    #                 print(f", lr {scheduler.get_last_lr()}")
    #                 train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
    #                 val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
                    
    #                 ax.set_xlim(0, config.NUM_EPOCHS + 2)
    #                 ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
                    
    #                 fig.canvas.draw()
    #                 fig.canvas.flush_events()
    #                 hdisplay.update(fig)
    #         else:
    #             for epoch in range(config.NUM_EPOCHS):
    #                 train_loss, train_acc = self.train_epoch(dataloaders['train'], criterion, optimizer, device)
                    
    #                 val_loss, val_acc = self.val_epoch(dataloaders['validate'], criterion, scheduler, device)
                    
    #                 history['train_loss'].append(train_loss)
    #                 history['train_acc'].append(train_acc)
    #                 history['val_loss'].append(val_loss)
    #                 history['val_acc'].append(val_acc)
                    
    #                 pbar_outer.update(1)
    #                 print(f"Epoch {epoch}: train_loss {train_loss}, train_iou {train_acc},  val_loss {val_loss}, val_iou {val_acc}")
    #                 train_loss_line.set_data(range(1, epoch + 2), history['train_loss'])
    #                 val_loss_line.set_data(range(1, epoch + 2), history['val_loss'])
                    
    #                 ax.set_xlim(0, config.NUM_EPOCHS + 2)
    #                 ax.set_ylim(0, max(max(history['train_loss']), max(history['val_loss'])) + 1)
                    
    #                 fig.canvas.draw()
    #                 fig.canvas.flush_events()
    #                 hdisplay.update(fig)
    #         return history


    # def predict(self, image, mask, device, show_full=False, show=False):
    #     assert not(show_full and not(show)), "Show can't be true, while show_full is mode is active"
    #     self.model.eval()  # Модель в режиме оценки

    #     # Transforming data to tensors
    #     image = image.to(device)
    #     mask = mask.to(device)
    #     image = unsqueeze(image, 0)
    #     mask = unsqueeze(mask, 0)
    #     with no_grad():        
                
    #         output = self.compute_outputs(image, mask)
    #         pred = argmax(output, dim=1).squeeze(0).cpu()  # Getting prediction
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
    #             masks = masks.to(device)
                
    #             outputs = self.model(pixel_values=images).logits
    #             outputs = F.interpolate(outputs, size=(masks.shape[-2:]), mode='bilinear', align_corners=False)
    #             loss = criterion(outputs, masks.squeeze())
    #             test_loss += loss.item() * images.size(0)
                
    #             predicted = argmax(outputs, dim=1)
    #             test_iou += get_iou(predicted, masks.squeeze())
    #             if detailed:
    #                 test_acc += get_acc(predicted, masks.squeeze())
    #                 test_prec += get_prec(predicted, masks.squeeze())
    #                 test_recall += get_recall(predicted, masks.squeeze())
    #                 test_f1 += get_f1(predicted, masks.squeeze())
    #                 test_dice += get_dice(predicted, masks.squeeze())

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
        
        