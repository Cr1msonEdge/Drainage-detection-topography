# `datasets/`
Здесь размещаются наборы данных, используемые для обучения и тестирования моделей.
Данные не включены в репозиторий из-за большого объёма. Вы можете создать их самостоятельно или использовать собственные `.npy`-файлы.

### Структура 
В репозитории включён демонстрационный датасет `example_dataset` для описания структуры хранения датасетов. 
Если у Вас есть изображения и маски в формате `.tif`, вы можете автоматически сформировать датасет с помощью встроенных функций:

```
from engine.application import create_patch_dataset_from_tif

create_patch_dataset_from_tif(
    image_dir='path/to/tif/images',
    mask_dir='path/to/tif/masks',
    output_dir='datasets/my_dataset',
    patch_size=256,
    stride=128
)
```

```
from helper.utilities import save_train_test_split

save_train_test_split(
    dataset_name='dataset_name', 
    images=images, 
    masks=masks, 
    val_size=0.1,
    test_size=0.1,
)
```