from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from robustbench.data import CORRUPTIONS, PREPROCESSINGS


def load_imagenet_o_64x64_c(
    n_examples: Optional[int] = 10000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 10000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'ImageNet-O-64x64-C' / corruptions[0] / str(severity)
    imagenet_o_64x64_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(imagenet_o_64x64_c) + 1
    repeated_imagenet_o_64x64_c = data.ConcatDataset([imagenet_o_64x64_c] * repeats)
    test_loader = data.DataLoader(repeated_imagenet_o_64x64_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test


def load_tiny_imagenet_c(
    n_examples: Optional[int] = 10000,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    prepr: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 10000 points.')

    assert len(
        corruptions
    ) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"
    # TODO: generalize this (although this would probably require writing a function similar to `load_corruptions_cifar`
    #  or alternatively creating yet another CustomImageFolder class that fetches images from multiple corruption types
    #  at once -- perhaps this is a cleaner solution)

    data_folder_path = Path(data_dir) / 'Tiny-ImageNet-C' / corruptions[0] / str(severity)
    tiny_imagenet_c = ImageFolder(data_folder_path, prepr)
    repeats = n_examples // len(tiny_imagenet_c) + 1
    repeated_tiny_imagenet_c = data.ConcatDataset([tiny_imagenet_c] * repeats)
    test_loader = data.DataLoader(repeated_tiny_imagenet_c,
                                  batch_size=n_examples,
                                  shuffle=shuffle,
                                  num_workers=2)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test


def load_imagenet_o_64x64(
    n_examples: Optional[int] = 10000,
    data_dir: str = './data',
    transforms_test: Callable = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_examples > 10000:
        raise ValueError(
            'The evaluation is currently possible on at most 5000 points-')

    imagenet = ImageFolder(data_dir + '/imagenet-o', transforms_test)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=n_examples,
                                  shuffle=True,
                                  num_workers=4)

    x_test, y_test = next(iter(test_loader))

    return x_test, y_test
