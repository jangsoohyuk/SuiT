# Modified for Adaptive Superpixels feature
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py.
Modifications include support for adaptive superpixel generation.
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from timm.loss import SoftTargetCrossEntropy
import utils

# Import for adaptive superpixel generation
from datasets import generate_superpixels, Denormalize 


def train_one_epoch(model: torch.nn.Module, criterion: SoftTargetCrossEntropy,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None,
                    # --- Parameters for Adaptive Superpixels ---
                    parameter_predictor: Optional[torch.nn.Module] = None, # The model to predict K and m
                    adaptive_superpixels: bool = False, # Flag to enable/disable adaptive mode
                    denormalize_transform: Optional[Denormalize] = None, # Transform to denormalize images for SLIC
                    args_spix_downsample: Optional[int] = None, # Downsampling factor for SLIC
                    args_spix_method: Optional[str] = None): # SLIC algorithm ('fastslic' or 'slic')
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    is_suit = 'suit' in args.model
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for data in metric_logger.log_every(data_loader, print_freq, header):
        samples = data[0].to(device, non_blocking=True)
        targets = data[-1].to(device, non_blocking=True) # Last element is always target according to SpixImageFolder
        
        spix_id = None # Initialize spix_id, it will be populated if model is SUIT
        if is_suit: # SUIT models require superpixel IDs (spix_id)
            if adaptive_superpixels:
                # Adaptive mode: Generate superpixels on-the-fly using the parameter_predictor
                # Ensure all necessary components for adaptive superpixel generation are provided
                assert parameter_predictor is not None, "ParameterPredictor module must be provided for adaptive superpixels."
                assert denormalize_transform is not None, "Denormalize transform must be provided for adaptive superpixels."
                assert args_spix_downsample is not None, "args.downsample (spix_downsample) must be provided for adaptive superpixels."
                assert args_spix_method is not None, "args.spix_method must be provided for adaptive superpixels."
                
                # Use the parameter_predictor to get K (number of segments) and m (compactness)
                with torch.no_grad(): # Predictor is usually not trained jointly in this setup, or handled by its own optimizer
                    pred_k, pred_m = parameter_predictor(samples) 
                
                # Generate superpixel assignments using the predicted K and m
                # The generate_superpixels function handles batch processing.
                spix_id = generate_superpixels(
                    image_tensor_batch=samples, 
                    k_values=pred_k.to(device), 
                    m_values=pred_m.to(device), 
                    denormalize_transform=denormalize_transform, 
                    downsample_factor=args_spix_downsample, 
                    spix_method=args_spix_method, 
                    device=device # Ensure assignments are on the correct device
                )
            elif len(data) == 3: 
                # Static mode for SUIT models: superpixels are pre-computed by the DataLoader (SpixImageFolder)
                # data format is expected to be (sample, spix_id, target)
                spix_id = data[1].to(device, non_blocking=True)
            # If not adaptive and len(data) is not 3 (e.g. for non-SUIT models or error), 
            # spix_id remains None. The model should handle spix_id=None if it's not a SUIT model.
            # If it's a SUIT model and spix_id is None here, it's likely an issue.

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            if is_suit:
                outputs = model(samples, spix_id)
            else:
                outputs = model(samples)

            if not args.cosub:
                loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad() # Disables gradient calculations during evaluation
def evaluate(data_loader, model, device, is_suit=True, 
             # --- Parameters for Adaptive Superpixels (consistency with train_one_epoch) ---
             adaptive_superpixels: bool = False, # Flag to enable/disable adaptive mode
             parameter_predictor: Optional[torch.nn.Module] = None, # Model to predict K and m
             denormalize_transform: Optional[Denormalize] = None, # Transform to denormalize images
             args_spix_downsample: Optional[int] = None, # Downsampling factor for SLIC
             args_spix_method: Optional[str] = None): # SLIC algorithm
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for data in metric_logger.log_every(data_loader, 10, header):
        images = data[0].to(device, non_blocking=True)
        target = data[-1].to(device, non_blocking=True) # Last element is always target
        
        spix_id = None # Initialize spix_id
        if is_suit: # SUIT models require superpixel IDs
            if adaptive_superpixels:
                # Adaptive mode for evaluation: Generate superpixels on-the-fly.
                # This ensures evaluation uses the same mechanism as training if adaptive.
                assert parameter_predictor is not None, "ParameterPredictor must be provided for adaptive superpixels in eval."
                assert denormalize_transform is not None, "Denormalize transform must be provided for adaptive superpixels in eval."
                assert args_spix_downsample is not None, "args.downsample (spix_downsample) must be provided for adaptive superpixels in eval."
                assert args_spix_method is not None, "args.spix_method must be provided for adaptive superpixels in eval."

                # Predict K and m using the parameter_predictor
                pred_k, pred_m = parameter_predictor(images) # `images` is the batch of samples
                
                # Generate superpixel assignments
                spix_id = generate_superpixels(
                    image_tensor_batch=images, 
                    k_values=pred_k.to(device), 
                    m_values=pred_m.to(device), 
                    denormalize_transform=denormalize_transform, 
                    downsample_factor=args_spix_downsample, 
                    spix_method=args_spix_method, 
                    device=device
                )
            elif len(data) == 3:
                # Static mode for SUIT models: superpixels are pre-computed by DataLoader
                spix_id = data[1].to(device, non_blocking=True)
            # If not adaptive and len(data) is not 3, spix_id remains None.

        # Compute model output
        with torch.cuda.amp.autocast():
            if is_suit:
                output = model(images, spix_id)
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
