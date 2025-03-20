from helper.dataobj import DrainageDataset
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader


DATASET_NAMES = [
    "dataset_hard"
]


def get_dataset_folder(name=None):
    # ? Probably add some default dataset in future
    assert name is not None, "Dataset name shouldn't be empty"
    curr_file = Path(__file__).resolve()  # 
    current_dir = curr_file.parent.parent  # Getting project directory
    datasets_dir = current_dir / 'datasets'
    
    datasets_lists = []
    if datasets_dir.exists():
        for fold in datasets_dir.iterdir():
            print(fold.name)
            datasets_lists.append(fold.name)
            
    if name not in datasets_lists:
        print(f"Dataset {name} not found.")
        return
    
    return datasets_dir / name
        
def get_dataloader(mode='train', name=None, device=None, batch_size=128, num_workers=4):
    assert mode in ['train', 'tests'], f"Mode {mode} is invalid."
    assert device in ['cpu', 'cuda', None], f"Device {device} is invalid."
        
        
    if name == None:
        name = DATASET_NAMES[0]
        
    data_dir = get_dataset_folder(name) / mode
    if data_dir.exists():
        # Checking if both images.npy and masks.npy exist
        data_folder_content = [i.name for i in data_dir.iterdir()]
        
        if "images.npy" in data_folder_content and "masks.npy" in data_folder_content:
            images = np.load(f"data_dir/{images.npy}")
            masks = np.load(f"data_dir/{masks.npy}")
            
            data = DrainageDataset(images=images, masks=masks, mode=mode)
            dataloader = DataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers)
            
            return dataloader

        else:
            print(f"Didn't find images.npy and masks.npy in Dataset:{name} folder.")
            return
    

if __name__ == "__main__":
    print(get_dataloader())
    