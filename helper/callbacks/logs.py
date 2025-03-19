import json
import pathlib
from helper.callbacks.metrics import get_acc, get_dice, get_f1, get_iou, get_prec, get_recall


# Saving paths
LOGS_PATH = str(pathlib.Path(__file__).parent.resolve()) + '/logs_results'
LOGS_HISTORY_PATH = str(pathlib.Path(__file__).parent.resolve()) + '/logs_hist_results'


def to_lists(data):
    """
    Transforms pairs of key - value to key - list of values (mostly np.array to list) for dict
    """
    for key, value in data.items():
        if not (isinstance(value, float) or (isinstance(value, int))):
            data[key] = list(value)
    return data


def get_metrics(preds, labels):
    """
    Gets metrics and returns dict of metrics 
    """
    return {
        'accuracy': get_acc(preds, labels),
        'dice': get_dice(preds, labels),
        'f1': get_f1(preds, labels),
        'iou': get_iou(preds, labels),
        'precision': get_prec(preds, labels),
        'recall': get_recall(preds, labels),
    }


def get_clear_metrics(metrics):
    """
    Gets metrics and returns dict of metrics from list
    """
    return {
        'accuracy': metrics['acc'],
        'f1': metrics['f1'],
        'recall': metrics['recall'],
        'precision': metrics['prec'],
        'iou': metrics['iou'],
        'dice': metrics['dice'],
        'loss': metrics['loss']
    }


def save_history_callback(model, history, hyperparams=None, name=None):
    """
    Saves history callback
    """
    data = {
        'train_loss': history['train_loss'],
        'train_iou': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_iou': history['val_acc']
    }
    
    model_dict = {
        'model_name': model.get_name(),
        'hyperparams': hyperparams
    }
    
    js = {
        'model': model_dict,
        'data': data
    }
    if name is None:
        with open(f'{LOGS_HISTORY_PATH}\\{model.get_json_name()}', 'w') as outFile:
            json.dump(js, outFile, indent=4)
    else:
        with open(f'{LOGS_HISTORY_PATH}\\{name}', 'w') as outFile:
            json.dump(js, outFile, indent=4)


def save_resulting_callback(model, metrics, hyperparams=None, name=None):
    '''
    Saves the test results to json files
    It is assumed that the file name has the format `model_name-iter.json`, where iter is an iteration of the test or model
    '''
    data = get_clear_metrics(metrics)
    
    js = {
        'model_name': model.get_name(),
        'hyperparams': hyperparams,
    }  
    pc = {
        'model': js,
        'data': data
    }
    
    if name is None:
        with open(f'{LOGS_PATH}\\{model.get_json_name()}', 'w') as outFile:
            json.dump(pc, outFile, indent=4)
    else:
        with open(f'{LOGS_PATH}\\{name}.json', 'w') as outFile:
            json.dump(pc, outFile, indent=4)
    
    