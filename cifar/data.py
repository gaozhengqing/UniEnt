from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets

from robustbench.data import _load_dataset, CORRUPTIONS, PREPROCESSINGS


def load_svhn(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.SVHN(root=data_dir,
                            split='test',
                            transform=transforms_test,
                            download=False)
    return _load_dataset(dataset, n_examples)


def load_svhn_c(
    n_examples: int,
    severity: int = 5,
    data_dir: str = './data',
    shuffle: bool = False,
    corruptions: Sequence[str] = CORRUPTIONS,
    _: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 1 <= severity <= 5
    n_total_svhn = 26032

    data_dir = Path(data_dir)
    data_root_dir = data_dir / 'SVHN-C'

    labels_path = data_root_dir / 'labels.npy'
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_svhn:severity *
                            n_total_svhn]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # Make it in the PyTorch format
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    # Make it compatible with our models
    x_test = x_test.astype(np.float32) / 255
    # Make sure that we get exactly n_examples but not a few samples more
    x_test = torch.tensor(x_test)[:n_examples]
    y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test
