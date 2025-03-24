import numpy as np


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
