

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset


def load_imagenet_from_directory(
    directory: str,
    transfoms: 'transforms',
    subset: int = None,
    batchsize: int = 32, shuffle: bool = False,
    require_label: bool = True, num_of_workers: int = 12
) -> torch.utils.data.DataLoader:
    dataset = datasets.ImageFolder(directory, transfoms)
    if subset:
        dataset = Subset(dataset, indices=[_ for _ in range(0, subset)])
    if require_label:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            drop_last=True
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_of_workers,
            pin_memory=False,
            collate_fn=lambda x: torch.cat(
                [sample[0].unsqueeze(0) for sample in x], dim=0),
            drop_last=False
        )
