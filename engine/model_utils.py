from torch import load, save
from pathlib import Path
import datetime


MODEL_NAMES = [
    "Unet",
    "UnetImageNet",
    "Unet++",
    "Segformer",
    "DeepLabV3"
    
]
def get_model_folder(name, verbose=-1):
    """
    Params:
    verbose: if -1, no prints are allowed. If 0, only errors are printed. If 1, everything is printed
    """
    assert name is not None, "Model name can't be None."
    assert name in MODEL_NAMES, "Model {name} not found in available models' list."
    
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
    
    
def save_model(model, run_id, verbose=-1):
    model_name = model.get_name()
    try:
        model_folder = get_model_folder(model_name)
    except:
        print("Got wrong model name or model is unknown. Returning")
        return
    
    file_name = f"{model_folder}-{run_id}.pt"
    save(model.state_dict(), file_name)
    
    if verbose != -1:
        print('Save completed.')
    

def load_state_dict(name):
    model_name = name[:name.find('-')]
    assert model_name in MODEL_NAMES, f"Model name {name} is not correct."
    try:
        model_folder = get_model_folder(model_name)
    except:
        print("Got wrong model name or model is unknown. Returning")
        return
    
    state_dict = load(model_folder / model_name)
    return state_dict


if __name__ == '__main__':
    print(get_model_folder('Unet', verbose=1))
    