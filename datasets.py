# Modified for Adaptive Superpixels feature
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


# Superpixel generation function
def generate_superpixels(
    image_tensor_batch: torch.Tensor, 
    k_values: torch.Tensor, 
    m_values: torch.Tensor, 
    denormalize_transform: Callable, 
    downsample_factor: int, 
    spix_method: str, 
    device: torch.device
) -> torch.Tensor:
    """
    Generates superpixel assignments for a batch of images.

    Args:
        image_tensor_batch (torch.Tensor): Batch of image tensors (N, C, H, W).
                                           Assumed to be normalized.
        k_values (torch.Tensor): Tensor of K values (number of superpixels) for each image (N,).
        m_values (torch.Tensor): Tensor of m values (compactness) for each image (N,).
        denormalize_transform (Callable): A callable transform (e.g., an instance of Denormalize)
                                          to convert normalized image tensors back to approx. [0, 255] range.
        downsample_factor (int): Factor by which to downsample the image before applying SLIC.
                                 Higher values speed up computation but reduce detail.
        spix_method (str): Superpixel algorithm to use, either 'fastslic' or 'slic'.
        device (torch.device): The Torch device to which the final assignment tensor should be moved.

    Returns:
        torch.Tensor: A batch of superpixel assignment maps. 
                      Shape (N, 1, H_spix, W_spix), where H_spix and W_spix are
                      the dimensions of the downsampled image used for SLIC.
    """
    batch_assignments = []
    for i in range(image_tensor_batch.shape[0]): # Iterate over each image in the batch
        img_tensor = image_tensor_batch[i] # Single image tensor (C, H, W)
        k_i = k_values[i].item() # K value for this specific image
        m_i = m_values[i].item() # m value for this specific image

        # Denormalize and prepare image for SLIC algorithm
        # SLIC typically expects an image in a standard format (e.g., uint8, [0,255])
        # Ensure tensor is on CPU for numpy conversion if it's not already.
        img_for_spix_normalized_cpu = img_tensor.cpu()
        img_for_spix_denormalized = denormalize_transform(img_for_spix_normalized_cpu)
        
        # Convert to NumPy array, scale to [0, 255], and change layout from (C,H,W) to (H,W,C)
        img_for_spix_numpy = np.array(img_for_spix_denormalized * 255).transpose(1, 2, 0)
        
        # Rescale (downsample) the image
        # anti_aliasing=True is important for quality when downscaling
        # channel_axis=2 indicates the last axis is channels for skimage.transform.rescale
        img_for_spix_rescaled = rescale(
            img_for_spix_numpy, 
            1 / downsample_factor, 
            anti_aliasing=True, 
            channel_axis=2
        ).round().clip(0, 255).astype(np.uint8)

        # Apply the chosen SLIC algorithm
        if spix_method == 'fastslic':
            slic_engine = SlicAvx2(num_components=int(k_i), compactness=m_i)
            assignment = slic_engine.iterate(img_for_spix_rescaled)
        elif spix_method == 'slic':
            # skimage.segmentation.slic expects HWC format.
            # channel_axis=-1 (or 2 for 3D HWC) might be needed depending on the version of skimage
            # and if it correctly infers multichannel from the last dimension.
            assignment = slic(
                img_for_spix_rescaled, 
                n_segments=int(k_i), 
                compactness=m_i, 
                channel_axis=-1, # Explicitly state channel axis for skimage
                start_label=0 # Ensures segments start from 0
            )
        else:
            raise NotImplementedError(f"Superpixel method {spix_method} not implemented.")

        # Convert assignment map to tensor and add a channel dimension (1, H_spix, W_spix)
        batch_assignments.append(torch.tensor(assignment, dtype=torch.int32).unsqueeze(0))

    # Stack all assignment maps into a single batch tensor (N, 1, H_spix, W_spix)
    # and move to the specified device.
    return torch.stack(batch_assignments).to(device)


class SpixImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        n_segments=196, # Default number of superpixels (K) if not adaptive
        compactness=10, # Default compactness (m) if not adaptive
        downsample=2,   # Default downsample factor for superpixel generation
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        spix_method = 'fastslic', # Default superpixel algorithm
        adaptive_superpixels = False, # If True, superpixels are not generated by this class
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.n_segments = n_segments # K for static superpixel mode
        self.adaptive_superpixels = adaptive_superpixels # Flag to control behavior in __getitem__
        # Denormalize transform is needed for superpixel generation as models operate on normalized images
        self.denormalize = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.compactness = compactness # m for static superpixel mode
        self.downsample = downsample
        self.spix_method = spix_method

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            tuple: 
                - If `self.adaptive_superpixels` is `True` (adaptive mode): 
                  `(sample, target)` 
                  where `sample` is the transformed image tensor, and `target` is the class index.
                - If `self.adaptive_superpixels` is `False` (static superpixel mode): 
                  `(sample, assignment, target)`
                  where `assignment` is the superpixel map tensor for the sample.
        """
        path, target = self.samples[index] # Get image path and class target
        sample = self.loader(path) # Load image using the specified loader

        # Apply image transformations (e.g., augmentations, normalization)
        if self.transform is not None:
            sample = self.transform(sample)

        # Apply target transformations if any
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.adaptive_superpixels:
            # In adaptive mode, superpixels are generated on-the-fly in the training loop (engine.py).
            # This class only returns the image and its target.
            return sample, target
        else:
            # In static mode, superpixels are generated here using fixed K (n_segments) and m (compactness).
            # The `generate_superpixels` function expects a batch of images,
            # so we unsqueeze the single sample to create a batch of 1.
            
            # Determine the device of the sample tensor. For datasets, this is typically CPU.
            # The generate_superpixels function handles moving the result to a specified device,
            # but its inputs (especially for denormalization and numpy conversion) are often expected on CPU.
            current_device = sample.device 

            # Call the helper function to generate superpixel assignments for the single image.
            # K and m values are taken from self.n_segments and self.compactness.
            assignment_batch = generate_superpixels(
                image_tensor_batch=sample.unsqueeze(0),  # Convert (C,H,W) to (1,C,H,W)
                k_values=torch.tensor([float(self.n_segments)], device=current_device), # K for this image
                m_values=torch.tensor([float(self.compactness)], device=current_device),# m for this image
                denormalize_transform=self.denormalize, # Pass the denormalization utility
                downsample_factor=self.downsample,     # Pass the downsampling factor
                spix_method=self.spix_method,          # Pass the SLIC algorithm choice
                device=current_device                  # Superpixels generated on current_device (CPU)
            )
            # Remove the batch dimension from the assignment map (1,1,H_spix,W_spix) -> (1,H_spix,W_spix)
            assignment = assignment_batch.squeeze(0) 
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
            # Pass adaptive_superpixels flag, defaulting to False for now as args.adaptive_superpixels is not yet defined
            # This will be updated later to use args.adaptive_superpixels
            dataset = SpixImageFolder(
                root, 
                transform=transform, 
                n_segments=args.n_spix_segments, 
                compactness=args.compactness, 
                downsample=args.downsample, 
                spix_method=args.spix_method,
                adaptive_superpixels=getattr(args, 'adaptive_superpixels', False) # Use getattr to avoid error if not present
            )
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
