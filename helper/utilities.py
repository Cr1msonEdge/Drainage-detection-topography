import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def calculate_mask_percentage(mask):
    """
    Calculate percentage of drainage system on the image
    """
    mask = mask.squeeze()
    mask = np.where(mask == 1, 1, 0)
    ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100
    return ratio


def get_bin(mask, max_bins=5, top_bin_percentage=11):
    """
    Return the number of bin for a mask
    
    Args:
    max_bins: the number of bins used for division
    top_bin_percentage: the masks that has percentage more than this parameters will have the bin number 'max_bins - 1'
    """
    mask = np.squeeze(mask)
    mask = np.where(mask == 1, 1, 0)  
    ratio = np.sum(mask) / mask.size * 100

    if ratio == 0:
        return 0

    if ratio >= top_bin_percentage:
        return max_bins - 1  

    step = top_bin_percentage / (max_bins - 2)  

    for i in range(1, max_bins - 1):
        if ratio < i * step:
            return i

    return max_bins - 2 


def save_train_test_split(dataset_name: str, images, masks, val_size=0.1, test_size=0.1, random_state=42, max_bins=5, top_percentage=11):
    assert len(images) == len(masks), "Size of images and masks are not equal."
    curr_file = Path(__file__)
    
    datasets_path = curr_file.parent.parent / 'datasets'
    save_dir = datasets_path / dataset_name

    # Create folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    temp = []
    train_dir = os.path.join(save_dir, 'train')
    temp.append(train_dir)

    if val_size:
        val_dir = os.path.join(save_dir, 'val')
        temp.append(val_dir)

    test_dir = os.path.join(save_dir, 'test')
    temp.append(test_dir)

    for d in temp:
        if not os.path.exists(d):
            os.makedirs(d)
    
    temp_ratio = val_size + test_size
    mask_bins = [get_bin(mask, max_bins=max_bins, top_bin_percentage=top_percentage) for mask in masks]
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        images, masks, test_size=temp_ratio, random_state=random_state, stratify=mask_bins
    )
    
    mask_bins = [get_bin(mask, max_bins=max_bins, top_bin_percentage=top_percentage) for mask in temp_masks]
    rel_test_ratio = test_size / temp_ratio if temp_ratio != 0 else 0
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=rel_test_ratio, random_state=random_state, stratify=mask_bins
    )
    
    np.save(os.path.join(train_dir, 'images.npy'), train_images)
    np.save(os.path.join(train_dir, 'masks.npy'), train_masks)

    if val_size:
        np.save(os.path.join(val_dir, 'images.npy'), val_images)
        np.save(os.path.join(val_dir, 'masks.npy'), val_masks)
    
    np.save(os.path.join(test_dir, 'images.npy'), test_images)
    np.save(os.path.join(test_dir, 'masks.npy'), test_masks)
    
    print(f"Successfully saved data in:\n  - {train_dir}\n  - {val_dir}\n  - {test_dir}")
