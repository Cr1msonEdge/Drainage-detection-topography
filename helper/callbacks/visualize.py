from helper.data.dataobj import DrainageDataset
from numpy import transpose, logical_and
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


LOGS_RESULTS_PATH = 'logs_results'
LOGS_HIST_PATH = 'logs_hist_results'


def draw_accuracy(model_name, data: dict, config, save=False, mode='accuracy'):
    """
    Draws history of accuracy or loss for model 
    """
    MODES = ['accuracy', 'loss']
    if mode not in MODES:
        print(f"Mode {mode} is not available.")
        return
    
    plt.figure(figsize=(16, 10))
    plt.title(f"Model {model_name} {mode} performance")
    ax = plt.gca()
    
    if mode == 'loss':
        n_epochs = len(data['train_loss'])
        plt.plot(range(1, n_epochs + 1), data['train_loss'], 'b', label=f"Train {mode}")
        plt.plot(range(1, n_epochs + 1), data['val_loss'], 'r', label=f"Val {mode}")
        
        ax.set_ylim(0, max(max(data['train_loss']), max(data['val_loss'])))
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
    elif mode == 'accuracy':
        n_epochs = len(data['train_iou'])
        #plt.plot(range(1, n_epochs + 1), data['train_iou'], linewidth=2)    
        plt.plot(range(1, n_epochs + 1), data['train_iou'], 'b', label=f"Train {mode}")
        plt.plot(range(1, n_epochs + 1), data['val_iou'], 'r', label=f"Val {mode}")
        ax.set_ylim(0, 1)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

    ax.set_xlim(0, len(data['train_iou']) + 2)
    
    plt.legend()
        
    hyperparameters_text = '\n'.join([f'{key}: {value}' for key, value in config.items()])
    plt.text(1, 1, 'Hyperparameters\n' + hyperparameters_text, 
            transform=plt.gca().transAxes, ha='left', va='center',
            bbox=dict(facecolor='white', alpha=1))
    save_path = './plots'

    
    if save:
        plt.savefig(save_path + model_name + '-' + mode)
        plt.show()
    else:
        plt.show()
    
    
def showimage(dataset: DrainageDataset, idx: int):
    """
    Showing image of Drainage Dataset
    """
    image, mask = dataset[idx]
    
    plt.subplot(1,2,1)
    plt.imshow(transpose(image, (1,2,0)))
    
    plt.axis('off')
    plt.title("IMAGE")

    plt.subplot(1,2,2)
    plt.imshow(transpose(mask, (1,2,0)), cmap='gray')

    plt.axis('off')
    plt.title("GROUND TRUTH")
    plt.show()


def show_prediction(image, pred, mask):
    # Transforming preds to numpy array
    pred = pred.numpy()
    
    # Creating the intersection of prediction and mask
    intersection = logical_and(pred, mask)

    # Showing images
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Image should be 256 x 256 x 
    axs[0][0].imshow(transpose(image, (1, 2, 0)))
    axs[0][0].set_title("Original Image")
    axs[0][0].axis('off')
    
    # Prediction, mask and intersection are 256 x 256
    axs[0][1].imshow(pred, cmap='gray')
    axs[0][1].set_title("Model Prediction")
    axs[0][1].axis('off')

    #print(f"mask: {mask.shape}")

    axs[1][0].imshow(mask, cmap='gray')
    axs[1][0].set_title("Ground Truth Mask")
    axs[1][0].axis('off')
    #print(f"inter: {intersection.shape}")

    axs[1][1].imshow(intersection, cmap='gray')
    axs[1][1].set_title("Intersection (Prediction & Mask)")
    axs[1][1].axis('off')

    plt.show()


def draw_metrics(model_name, metrics, hyperparams):
    metric_names = list(metrics.keys())
    metric_values = [float(value) for value in metrics.values()]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric_names, y=metric_values, palette='Paired')

    plt.title(f'Model {model_name} Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')

    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
    
    hyperparameters_text = '\n'.join([f'{key}: {value}' for key, value in hyperparams.items()])

    # Adding hyperparams to text
    plt.text(1, 1, 'Hyperparameters\n' + hyperparameters_text, 
            transform=plt.gca().transAxes, ha='left', va='center',
            bbox=dict(facecolor='white', alpha=1))
    plt.show()
    
    
def draw_resulting_log(logname):
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    # Path to logs
    log_path = os.path.join(script_dir, LOGS_RESULTS_PATH, logname)
    
    with open(log_path) as file:
        data = json.load(file)
    draw_metrics(data['model']['model_name'], data['data'], data['model']['hyperparams'])
    
    
def draw_history_log(logname, mode='accuracy'):
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    log_path = os.path.join(script_dir, LOGS_HIST_PATH, logname)
    
    with open(log_path) as file:
        data = json.load(file)
    draw_accuracy(data['model']['model_name'], data['data'], data['model']['hyperparams'], False, mode=mode)    
    

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def compare_models_sns_barplot(models, metrics_dict, hyperparams_dict):
    """
    Args:
        models (list): List of models names (напр. ['Model A', 'Model B']).
        metrics_dict (dict): Dict of metrics for each model. 
                             Key — model name, value — dict of metrics.
                             E.g: {'Model A': {'accuracy': 0.95, 'precision': 0.92}, 'Model B': {...}}
        hyperparams_dict (dict): Dict of hyperparams for models. 
                                Key — model name, value — dict of hyperparams.
                                 E.g: {'Model A': {'learning_rate': 0.01, 'epochs': 10}, 'Model B': {...}}
    """
    data = []
    
    for model in models:
        for metric, value in metrics_dict[model].items():
            data.append({'Model': model, 'Metric': metric, 'Value': value})
    
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))

    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df, palette='Paired')

    plt.title('Models comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    
    # Adding values above columns
    for p in ax.patches:
        value = p.get_height()
        if value > 0:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', 
                        fontsize=8, color='black', xytext=(0, 3), 
                        textcoords='offset points')

    # Showing hyperparams
    hyperparams_texts = [f"{model}: " + ', '.join([f"{key}: {value}" for key, value in hyperparams_dict[model].items()]) for model in models]
    hyperparams_text = '\n'.join(hyperparams_texts)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.figtext(1, -0.1, hyperparams_text, horizontalalignment='right')

    plt.show()


def show_prediction(image, pred, mask, show_intersection=False):
    # Transform the prediction and mask to numpy
    pred = pred.numpy()
    if intersection:
        if not show_intersection:
            # Showing original image, mask, prediction 
            intersection = logical_and(pred, mask)
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
            
            axs[0][0].imshow(transpose(image, (1, 2, 0)))
            axs[0][0].set_title("Original Image")
            axs[0][0].axis('off')
            
            # Prediction, mask and intersection are 256 x 256
            axs[0][1].imshow(pred, cmap='gray')
            axs[0][1].set_title("Model Prediction")
            axs[0][1].axis('off')

            print(f"mask: {mask.shape}")

            axs[0][2].imshow(mask, cmap='gray')
            axs[0][2].set_title("Ground Truth Mask")
            axs[0][2].axis('off')
            print(f"inter: {intersection.shape}")
        
        else:
            # Showing original image, mask, prediction and intersection 
            # Creating the intersection of the mask and
            intersection = logical_and(pred, mask)

            # Showing images
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

            # Image should be 256 x 256 x 3
            axs[0][0].imshow(transpose(image, (1, 2, 0)))
            axs[0][0].set_title("Original Image")
            axs[0][0].axis('off')
            
            # Prediction, mask and intersection are 256 x 256
            axs[0][1].imshow(pred, cmap='gray')
            axs[0][1].set_title("Model Prediction")
            axs[0][1].axis('off')

            print(f"mask: {mask.shape}")

            axs[1][0].imshow(mask, cmap='gray')
            axs[1][0].set_title("Ground Truth Mask")
            axs[1][0].axis('off')
            print(f"inter: {intersection.shape}")

            axs[1][1].imshow(intersection, cmap='gray')
            axs[1][1].set_title("Intersection (Prediction & Mask)")
            axs[1][1].axis('off')
    
    else:
        # Showing prediction only
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        axs[0][0].imshow(pred, cmap='gray')
        axs[0][0].set_title("Model Prediction")
        axs[0][0].axis('off')
        
    plt.show()


def show_predictions(image, preds: list, models_names: list):
    """
    Shows predictions of models 
    """
    assert len(preds) == len(models_names), "Number of predictions and models names should be equal."
    ncols = 2
    nrows = len(preds) // 2 + len(preds) % 2
    print(nrows, ncols, "goida")
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    
    axs[0][0].imshow(transpose(image, (1, 2, 0)))
    axs[0][0].set_title("Original Image")
    axs[0][0].axis('off')
    
    for i in range(len(preds)):
        axs[(i + 1) // 2][(i + 1) % 2].imshow(preds[i], cmap='gray')
        axs[(i + 1) // 2][(i + 1) % 2].set_title(models_names[i])
        axs[(i + 1) // 2][(i + 1) % 2].axis('off')
        