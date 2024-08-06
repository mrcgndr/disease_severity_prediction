"""
Module for handling single plant image datasets
"""

from typing import Dict, List, Optional, Tuple

import albumentations as al
import numpy as np
import torch
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from . import HDF5Dataset
from .augmentation import ImageAugmentationPipeline
from .norm import BaseNormalizer
from .utils import DistributedImbalancedDatasetSampler, ImbalancedDatasetSampler, balanced_train_val_split

DEFAULT_AUG_PIPELINE = al.Compose(
    [
        al.Flip(p=0.5),
        al.Rotate(p=0.25, interpolation=1, border_mode=4),
    ]
)

DATASET_MEAN = [0.03560187, 0.08843552, 0.055837303, 0.199051, 0.4454364]
DATASET_STD = [0.052432425, 0.086994074, 0.07697101, 0.15138984, 0.2818574]


class PlantImageData(HDF5Dataset):
    """Class for single plant image datasets

    Dataset(s) has/have to be provided as HDF5 file
    with following structure:

    /image: Images (N x C x H x W)
    /valid: Valid mask (N x H x W)
    /image_date: Image recording date (N)
    /annotation_date: Image annotation date (N)
    /ds: Disease severity (DS) value of plant in image (N)
    /das: Days after sowing (N)
    /dac: Days after canopy closure (N)
    /npg: Number of possible Cercospora generations (N)
    /gdd: Cumulative growing degree days (N)

    optional
    /seg_mask: Segmentation masks (N x H x W)
    /cover_ratio: Cover ratio (N)
    /vi_hist: Histogram of pixel VI information (N x 201)
    """

    all_image_channels = np.array(["B", "G", "R", "REDGE", "NIR"])

    def __init__(
        self,
        h5paths: List[str],
        labels: List[str] = ["ds", "dac"],
        augmentations: Optional[al.BaseCompose] = None,
        normalizer: Optional[BaseNormalizer] = None,
        image_channels: List[str] = ["B", "G", "R", "REDGE", "NIR"],
        **kwargs,
    ):
        super().__init__(paths=h5paths, data_key="image")
        self.normalizer = normalizer
        self.augmentations = augmentations
        self.labels = labels
        if np.sum(self.lengths) == 0:
            raise ValueError("Given data paths result in empty dataset.")
        for channel in image_channels:
            if channel not in self.all_image_channels:
                raise ValueError(f"Unknown channel name '{channel}'. Available channels: {self.all_image_channels}.")
        self.selected_channels = [np.argmax(self.all_image_channels == channel) for channel in image_channels]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        raw = super().__getitem__(idx)

        img = (np.asarray(raw["image"])[self.selected_channels]).astype(float) / 32768.0
        labels = [torch.tensor(raw[label], dtype=torch.float32) for label in self.labels]
        valid = raw["valid"]

        target = {
            "dataset": raw["dataset"],
            "image_date": raw["image_date"],
            "annotation_date": raw["annotation_date"],
            "file_image_id": torch.as_tensor(raw["file_idx"]),
            "total_image_id": torch.as_tensor(raw["total_idx"]),
            "labels": labels,
            "valid": valid,
        }

        # augment
        if self.augmentations:
            aug = self.augmentations(image=np.transpose(img, axes=(1, 2, 0)), mask=valid.astype(int))

            img = np.transpose(aug["image"], axes=(2, 0, 1))
            valid = aug["mask"].astype(bool)

        # normalize, if normalizer given
        if self.normalizer is not None:
            img = self.normalizer(batch=img[None], valid_mask=valid[None])[-1]

        return torch.tensor(img, dtype=torch.float), target


class PlantImageDataModule(LightningDataModule):
    """Data module for single plant image datasets."""

    def __init__(
        self,
        h5paths: List[str],
        balanced_split_label: Optional[List[str]] = None,
        balanced_train_label: Optional[List[str]] = None,
        batch_size: int = 32,
        labels: List[str] = ["ds", "dac"],
        augmentations: Optional[al.BaseCompose] = None,
        normalizer: Optional[BaseNormalizer] = None,
        image_channels: List[str] = ["B", "G", "R", "REDGE", "NIR"],
        val_size: float = 0.25,
        distributed: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.h5paths = h5paths
        self.balanced_split_label = balanced_split_label
        self.balanced_train_label = balanced_train_label
        if augmentations:
            self.augmentations = augmentations
        else:
            self.augmentations = ImageAugmentationPipeline({})
        self.normalizer = normalizer
        self.labels = labels
        self.image_channels = image_channels
        self.val_size = val_size
        self.distributed = distributed
        self.batch_size = batch_size
        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": kwargs.get("pin_memory", False),
            "persistent_workers": kwargs.get("persistent_workers", False),
            "drop_last": True,
        }
        if seed:
            seed_everything(seed, workers=True)

        test_h5paths = kwargs.get("test_h5paths", None)
        if test_h5paths:
            self.test_h5paths = test_h5paths

        if val_size > 0:
            self.train_indices, self.val_indices = balanced_train_val_split(
                PlantImageData(
                    h5paths=self.h5paths,
                    labels=self.labels,
                    image_channels=self.image_channels,
                ),
                balance_by=self.balanced_split_label,
                val_size=self.val_size,
                seed=seed,
            )
            self.train_dataset = Subset(
                PlantImageData(
                    h5paths=self.h5paths,
                    augmentations=self.augmentations["train"],
                    normalizer=self.normalizer,
                    labels=self.labels,
                    image_channels=self.image_channels,
                ),
                indices=self.train_indices,
            )
            self.val_dataset = Subset(
                PlantImageData(
                    h5paths=self.h5paths,
                    augmentations=self.augmentations["validate"],
                    normalizer=self.normalizer,
                    labels=self.labels,
                    image_channels=self.image_channels,
                ),
                indices=self.val_indices,
            )
            if self.balanced_train_label:
                targets = [
                    torch.as_tensor(self.train_dataset.dataset.get_all(b_label)) for b_label in self.balanced_train_label
                ]
                self.train_targets = [target[self.train_indices] for target in targets]
                self.val_targets = [target[self.val_indices] for target in targets]
        else:
            self.train_dataset = PlantImageData(
                h5paths=self.h5paths,
                augmentations=self.augmentations["train"],
                normalizer=self.normalizer,
                labels=self.labels,
                image_channels=self.image_channels,
            )
            self.train_indices = torch.arange(len(self.train_dataset))
            if self.balanced_train_label:
                self.train_targets = [
                    torch.as_tensor(self.train_dataset.get_all(b_label)) for b_label in self.balanced_train_label
                ]
            self.val_dataset = None

        if test_h5paths:
            self.test_dataset = PlantImageData(
                h5paths=self.test_h5paths,
                augmentations=self.augmentations["test"],
                normalizer=self.normalizer,
                labels=self.labels,
                image_channels=self.image_channels,
            )
            if self.balanced_train_label:
                self.test_targets = [
                    torch.as_tensor(self.test_dataset.get_all(b_label)) for b_label in self.balanced_train_label
                ]
        else:
            self.test_dataset = None

    def train_dataloader(self) -> DataLoader:
        if self.balanced_train_label:
            if self.distributed:
                sampler = DistributedImbalancedDatasetSampler(targets=self.train_targets)
            else:
                sampler = ImbalancedDatasetSampler(targets=self.train_targets)
            return DataLoader(dataset=self.train_dataset, sampler=sampler, **self.dataloader_kwargs)
        else:
            return DataLoader(
                dataset=self.train_dataset,
                shuffle=not self.distributed,
                sampler=DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None,
                **self.dataloader_kwargs,
            )

    def val_dataloader(self, shuffle: bool = False) -> List[DataLoader]:
        dataloaders = []
        if self.val_dataset:
            if self.balanced_train_label:
                if self.distributed:
                    sampler = DistributedImbalancedDatasetSampler(targets=self.val_targets)
                else:
                    sampler = ImbalancedDatasetSampler(targets=self.val_targets)
                dataloaders.append(DataLoader(dataset=self.val_dataset, sampler=sampler, **self.dataloader_kwargs))
            else:
                dataloaders.append(
                    DataLoader(
                        dataset=self.val_dataset,
                        shuffle=shuffle,
                        sampler=DistributedSampler(self.val_dataset) if self.distributed else None,
                        **self.dataloader_kwargs,
                    )
                )
        test_dl = self.test_dataloader()
        if test_dl is not None:
            dataloaders.append(test_dl)

        return dataloaders

    def test_dataloader(self, shuffle: bool = False) -> DataLoader:
        if self.test_dataset:
            # if self.balanced_train_label:
            #    if self.distributed:
            #        sampler = DistributedImbalancedDatasetSampler(targets=self.test_targets)
            #    else:
            #        sampler = ImbalancedDatasetSampler(targets=self.test_targets)
            #    return DataLoader(dataset=self.test_dataset, sampler=sampler, **self.dataloader_kwargs)
            # else:
            return DataLoader(
                dataset=self.test_dataset,
                shuffle=shuffle,
                sampler=DistributedSampler(self.test_dataset) if self.distributed else None,
                **self.dataloader_kwargs,
            )
        else:
            return None
