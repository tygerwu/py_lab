import os
import numpy as np
from typing import Any, Callable, Optional
from torchvision.datasets import VisionDataset
from detection.yolov5.data_utils import load_and_preprocess
from torch.utils.data import DataLoader


class YoloCocoDataset(VisionDataset):
    def __init__(
        self,
        img_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(img_dir, transforms, transform, target_transform)
        self.img_names = [img_name for img_name in os.listdir(
            img_dir) if img_name.endswith('jpg')]
        assert len(self.img_names) > 0, "Empty imgs"

    def __getitem__(self, index: int) -> Any:
        img_name = self.img_names[index]
        img_id = int(img_name.split('.')[0])
        img_path = os.path.join(self.root, img_name)
        img, cur_hw, ori_hw = load_and_preprocess(img_path=img_path)
        if self.transform is not None:
            img = self.transform(img)
        return [img, img_id, cur_hw, ori_hw]

    def __len__(self) -> int:
        return len(self.img_names)


def collate_img(samples):
    imgs = [sample[0] for sample in samples]
    return np.concatenate(imgs, axis=0)


def collate_all(samples):
    imgs = []
    img_ids = []
    ori_hws = []
    cur_hws = []
    for sample in samples:
        imgs.append(sample[0])
        img_ids.append(sample[1])
        cur_hws.append(sample[2])
        ori_hws.append(sample[3])

    imgs = np.concatenate(imgs, axis=0)
    ori_hws = np.concatenate(ori_hws, axis=0)
    cur_hws = np.concatenate(cur_hws, axis=0)
    return [imgs, img_ids, cur_hws, ori_hws]


def coco_dataloader(img_dir: str,
                    only_img: bool = True,
                    batch_size: int = 1,
                    workers: int = 1,
                    drop_last: bool = True,
                    pin_memory: bool = False
                    ):
    collate = collate_img if only_img else collate_all

    coco_dataset = YoloCocoDataset(img_dir=img_dir)

    return DataLoader(
        batch_size=batch_size,
        num_workers=workers,
        drop_last=drop_last,
        collate_fn=collate,
        pin_memory=pin_memory,
        dataset=coco_dataset
    )
