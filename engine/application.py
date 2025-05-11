import numpy as np
import rasterio
import os
import torch
from helper.callbacks.metrics import get_iou, get_acc, get_prec, get_recall, get_dice, get_f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm


def create_patch_dataset_from_tif(image_path: str, mask_path: str, name: str, patch_size: int = 256, thresh: float = 0.3, stride=128):
    """
    Save images' and masks' patches into numpy arrays.
    
    Args:
    image_path: filepath to image
    mask_path: filepath to mask
    name: name for dataset
    patch_size: size of a patch. Default: 256
    thresh: threshold for null data
    stride: stride for overlapping patches
    """
    dataset_dir = Path(__file__).resolve().parent.parent / 'datasets'
    output_dir = dataset_dir / name
    os.makedirs(output_dir, exist_ok=True)

    valid_indices = []
    image_patches = []

    # Read patches of image
    with rasterio.open(image_path) as img_src:
        img_width = img_src.width
        img_height = img_src.height

        for top in tqdm(range(0, img_height - patch_size + 1, stride), desc="Checking image patches"):
            for left in range(0, img_width - patch_size + 1, stride):
                window = Window(left, top, patch_size, patch_size)

                img = img_src.read(window=window)  # shape: (4, H, W)
                img = np.transpose(img, (1, 2, 0))  # → (H, W, C)

                rgb = img[..., :3]
                non_black_ratio = np.count_nonzero(rgb) / (patch_size * patch_size * 3)

                if non_black_ratio >= thresh:
                    valid_indices.append((left, top))
                    image_patches.append(img)

    image_patches = np.stack(image_patches)

    # Read masks which indexes are saved
    mask_patches = []
    with rasterio.open(mask_path) as mask_src:
        for (left, top) in tqdm(valid_indices, desc="Extracting mask patches"):
            window = Window(left, top, patch_size, patch_size)
            mask = mask_src.read(1, window=window)
            mask_patches.append(mask)

    mask_patches = np.stack(mask_patches)
    assert len(image_patches) == len(mask_patches), "Error when creating dataset. Number of images and masks is not the same"

    np.save(os.path.join(output_dir, "images.npy"), image_patches)
    np.save(os.path.join(output_dir, "masks.npy"), mask_patches)

    print(f"Saved {len(image_patches)} patches in {output_dir}")


def get_tif_metric(label_image: str, pred_image: str, verbose=True):
    """
    Return metrics of predicted tif file.
    
    Params:
    label_image: path to ground truth image
    source_image: path to predicted image
    """
    
    with rasterio.open(pred_image) as src:
        pred = torch.tensor(src.read(1), dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    
    with rasterio.open(label_image) as src:
        label = torch.tensor(src.read(1), dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    
    metrics = {
        'accuracy': get_acc(pred, label),
        'precision': get_prec(pred, label),
        'recall': get_recall(pred, label),
        'f1': get_f1(pred, label),
        'iou': get_iou(pred, label),
        'dice': get_dice(pred, label)
    }
    
    if verbose:
        print("=== Got result: ===")
        for key, value in metrics.items():
            print(f"{key} - {value:.2f}")
    
    return metrics
    