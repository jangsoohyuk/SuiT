# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from typing import Any, Callable, Optional
import os
import json

import numpy as np
import torch

from skimage.transform import rescale
from skimage.segmentation import slic
from fast_slic.avx2 import SlicAvx2

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class SpixImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        n_segments=196,
        compactness=10,
        downsample=2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        spix_method = 'fastslic',
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.n_segments = n_segments
        self.denormalize = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.compactness = compactness
        self.downsample = downsample
        self.spix_method = spix_method

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, assignment, target) where assignment is a map of superpixel indices for each pixel, and target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # augmented sample
            sample = self.transform(sample)
            
            # temporarily convert to [0, 255] and resize to acquire superpixels.
            sample_for_spix = np.array(self.denormalize(sample) * 255).transpose(1, 2, 0)
            sample_for_spix = rescale(sample_for_spix, 1 / self.downsample, anti_aliasing=True, channel_axis=2).round().clip(0, 255).astype(np.uint8)
            if self.spix_method == 'fastslic':
                slic_ = SlicAvx2(num_components=self.n_segments, compactness=self.compactness)
                assignment = slic_.iterate(sample_for_spix)
            elif self.spix_method == 'slic':
                assignment = slic(sample_for_spix, n_segments=self.n_segments, compactness=self.compactness)
            else:
                raise NotImplementedError

            assignment = torch.tensor(assignment).unsqueeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, assignment, target


class Denormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if 'suit' in args.model:
            dataset = SpixImageFolder(root, transform=transform, n_segments=args.n_spix_segments, compactness=args.compactness, downsample=args.downsample, spix_method=args.spix_method)
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
