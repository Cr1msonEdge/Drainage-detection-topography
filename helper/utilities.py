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


def save_train_test_split(dataset_name: str, save_dir_train: str, save_dir_test: str, images, masks, test=True, test_size=0.2, verbose=-1):
    assert len(images) == len(masks), "Size of images and masks are not equal."
    curr_file = Path(__file__)
    datasets_path = curr_file.parent.parent / 'datasets'
    dataset_name = datasets_path / dataset_name

    # Создаем папку для сохранения, если она не существует
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    # Делим изображения и маски на тренировочный и тестовый наборы
    train_images, test_images, train_masks, test_masks = train_test_split(
        images, masks, test_size=test_size, random_state=42
    )

    if verbose > 0:
        print(f"Number of train images: {len(train_images)}, of test images: {len(test_images)}, test_size={test_size}")

    # Сохраняем тренировочные данные
    np.save(os.path.join(dataset_name / 'train/' , '/train_images.npy'), train_images)
    np.save(os.path.join(dataset_name / 'train/', '/train_masks.npy'), train_masks)

    if test:
        np.save(os.path.join(dataset_name / 'test/', 'test_images.npy'), test_images)
        np.save(os.path.join(dataset_name / 'test/', 'test_masks.npy'), test_masks)
    else:
        np.save(os.path.join(dataset_name / 'val/', 'val_images.npy'), test_images)
        np.save(os.path.join(dataset_name / 'val/', 'val_masks.npy'), test_masks)

    if verbose > -1:
        print(f"Data successfully saved to {dataset_name}")
