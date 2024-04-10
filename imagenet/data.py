from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.utils.data as data
from torchvision.datasets import ImageFolder

from robustbench.data import CORRUPTIONS, PREPROCESSINGS


def load_imagenet_o_c(
    n_examples: Optional[int] = 5000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 5000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'ImageNet-O-C' / corruptions[0] / str(severity)
    imagenet_o_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(imagenet_o_c) + 1
    repeated_imagenet_o_c = data.ConcatDataset([imagenet_o_c] * repeats)
    test_loader = data.DataLoader(repeated_imagenet_o_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test
