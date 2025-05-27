from torch import load, save
from pathlib import Path
import uuid


MODEL_NAMES = [
    "Unet",
    "UnetImageNet",
    "Unet++",
    "Segformer",
    "DeepLabV3"
]


def get_model_folder(name, verbose=-1):
    """
    Return model's folder
    
    Params:
    verbose: if -1, no prints are allowed. If 0, only errors are printed. If 1, everything is printed
    """
    assert name is not None, "Model name can't be None."
    assert name in MODEL_NAMES, f"Model {name} not found in available models' list."
    
    curr_file = Path(__file__).resolve()
    saved_models_dir = curr_file.parent.parent / 'helper' / 'models' / 'saved'
    
    if not saved_models_dir.exists():
        if verbose != -1:
            print("There are problems with directory. Change path.")
        return
    
    model_folder = saved_models_dir / name
    if not model_folder.exists():
        if verbose == 1:
            print(f"Folder for model {name} is not found. Creating.")
        model_folder.mkdir(parents=True, exist_ok=True)
    
    else:
        if verbose == 1:
            print(f"Folder for model {name} is found.")
    
    return model_folder 
    

def get_model_file_path(name, verbose=-1):
    """
    Get full path to model's checkpoint file
    
    It is assumed that name is written in the form: "model-uid.ckpt".
    """
    assert name is not None, "Model name can't be None."
    assert name[:name.find('-')] in MODEL_NAMES, "Model {name} not found in available models' list."
    
    curr_file = Path(__file__).resolve()
    saved_models_dir = curr_file.parent.parent / 'helper' / 'models' / 'saved' / name[:name.find('-')]
    
    if not saved_models_dir.exists():
        if verbose != -1:
            raise Exception(f"Model {name} is not found. Model's {name} folder is not found.")

    
    models_path = saved_models_dir / name
    print(f"{models_path} asd")
    if models_path.exists():
        if verbose != -1:
            print(f"Successfully found model {name}.")
        return models_path
    else:
        raise Exception(f"Model {name} is not found.")


if __name__ == '__main__':
    print(get_model_file_path('Unet-12234.ckpt', verbose=1))
    