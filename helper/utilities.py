import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def calculate_mask_percentage(mask):
    """
    Calculate percentage of drainage system on the image
    """
    mask = mask.squeeze()
    mask = np.where(mask == 255, 1, 0)
    ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
    return ratio


def get_bin(mask, max_bins=5, top_bin_percentage=20):
    """
    Return the number of bin for a mask
    
    Params:
    max_bins: the number of bins used for division
    top_bin_percentage: the masks that has percentage more than this parameters will have the bin number 'max_bins - 1'
    """
    mask = mask.squeeze()
    mask = np.where(mask == 255, 1, 0)
    ratio = np.sum(mask, axis=(0, 1)) / (mask.shape[0] * mask.shape[1]) * 100
    if ratio == 0:
        return 0
    
    result = int(ratio)
    step = top_bin_percentage // (max_bins - 1) + 1
    
    for i in range(1, max_bins):
        if result in range(step * (i - 1), step * i):
            return i
    
    return max_bins - 1


def save_train_test_split(dataset_name: str, images, masks, val_size=0.1, test_size=0.1, random_state=42):
    assert len(images) == len(masks), "Size of images and masks are not equal."
    curr_file = Path(__file__)
    
    datasets_path = curr_file.parent.parent / 'datasets'
    save_dir = datasets_path / dataset_name

    # Create folder if it doesn't exist
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    
    train_dir = os.path.join(save_dir, 'train')
    if val_size:
        val_dir = os.path.join(save_dir, 'val')
    test_dir = os.path.join(save_dir, 'test')
    
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    temp_ratio = val_size + test_size
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        images, masks, test_size=temp_ratio, random_state=random_state
    )
    
    rel_test_ratio = test_size / temp_ratio if temp_ratio != 0 else 0
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=rel_test_ratio, random_state=random_state
    )
    
    np.save(os.path.join(train_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(train_dir, 'train_masks.npy'), train_masks)

    if val_size:
        np.save(os.path.join(val_dir, 'val_images.npy'), val_images)
        np.save(os.path.join(val_dir, 'val_masks.npy'), val_masks)
    
    np.save(os.path.join(test_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(test_dir, 'test_masks.npy'), test_masks)
    
    print(f"Данные успешно сохранены в каталогах:\n  - {train_dir}\n  - {val_dir}\n  - {test_dir}")
