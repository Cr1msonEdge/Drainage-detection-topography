from torch import sum, isnan, tensor, is_tensor
import segmentation_models_pytorch as smp
import lightning.pytorch as pl


class NeptuneCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Получаем агрегированные метрики из trainer.callback_metrics
        metrics = trainer.callback_metrics
        if 'avg_train_loss' in metrics:
            # Приводим к float, если необходимо, и указываем step=текущая эпоха
            trainer.logger.experiment["train/avg_loss"].append(
                metrics['avg_train_loss'].item(), step=trainer.current_epoch
            )
        if 'avg_train_metric' in metrics:
            trainer.logger.experiment["train/avg_metric"].append(
                metrics['avg_train_metric'].item(), step=trainer.current_epoch
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'avg_val_loss' in metrics:
            trainer.logger.experiment["val/avg_loss"].append(
                metrics['avg_val_loss'].item(), step=trainer.current_epoch
            )
        if 'avg_val_metric' in metrics:
            trainer.logger.experiment["val/avg_metric"].append(
                metrics['avg_val_metric'].item(), step=trainer.current_epoch
            )

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        keys = ['avg_test_loss', 'avg_test_accuracy', 'avg_test_precision',
                'avg_test_recall', 'avg_test_f1', 'avg_test_iou', 'avg_test_dice']
        for key in keys:
            if key in metrics:
                trainer.logger.experiment[f"test/{key}"].append(
                    metrics[key].item(), step=trainer.current_epoch
                )


def dice_score(preds, targets, smooth=1e-6):
    """
    Вычисление Dice Score для батча предсказаний и целевых значений (бинарные маски).
    
    Args:
        preds (torch.Tensor): Предсказания модели, размер (64, 256, 256)
        targets (torch.Tensor): Истинные маски, размер (64, 256, 256)
        smooth (float): Маленькое значение для избежания деления на ноль.
        
    Returns:
        float: Среднее значение Dice Score для всего батча.
    """
    
    # Убедимся, что оба тензора бинарны (0 или 1)
    
    # Вычисляем Dice Score для каждого изображения в батче
    intersection = sum(preds * targets, dim=(1, 2))  # Пересечение (предсказание и истина = 1)
    union = sum(preds, dim=(1, 2)) + sum(targets, dim=(1, 2))  # Сумма всех единиц в предсказаниях и истинных значениях
    
    dice = (2 * intersection + smooth) / (union + smooth)  # Dice Score для каждого изображения
    return dice.mean()  # Возвращаем среднее значение по батчу


def get_dice(pred, label):
    return dice_score(pred, label)
    
def get_acc(pred, label):
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', threshold=0.5)
    return smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')

def get_prec(pred, label):
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', threshold=0.5)
    result = smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
    if isnan(result):
        return tensor(0.0)
    return result


def get_recall(pred, label):
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', threshold=0.5)
    return smp.metrics.recall(tp, fp, fn, tn, reduction='micro')


def get_f1(pred, label):
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', threshold=0.5)
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')


def get_iou(pred, label):
    # print(pred)
    # print(label)
    tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='binary')

    #print(f'batchatillon {smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')}')
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')


# ! Probably shouldn't be used
class Metrics:
    def __init__(self, pred=None, label=None):
        if pred != None and label != None:
            tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', thershold=0.5)
            
            self.acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
            self.precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
            self.recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
            self.f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
            self.iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
            self.dice = dice_score(pred, label, num_classes=2, average='micro')
        
        elif pred != None and label == None or pred == None and label != None:
            print("When evaluating metrics some of tensors is None.")
            return
            
        else: 
            self.acc = 0
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.iou = 0
            self.dice =0
        
        
    def update(self, pred, label):
        if pred == None or label == None:
            print("When evaluating metrics some of tensors is None.")
            return
        
        tp, fp, fn, tn = smp.metrics.get_stats(pred, label, mode='multilabel', thershold=0.5)
        self.acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro')
        self.precision = smp.metrics.precision(tp, fp, fn, tn, reduction='micro')
        self.recall = smp.metrics.recall(tp, fp, fn, tn, reduction='micro')
        self.f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')
        self.iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        self.dice = dice_score(pred, label, num_classes=2, average='micro')
    
        return {
            'acc': self.acc,
            'prec': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'iou': self.iou,
            'dice': self.dice
        }
    
    def __str__(self) -> str:
        return f"acc: {self.acc}, prec: {self.precision}, recall: {self.recall}, f1: {self.f1}, iou: {self.iou}, dice: {self.dice}"
    