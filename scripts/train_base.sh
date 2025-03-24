#!/bin/bash

torchrun --master_port=29501 \
--nproc_per_node=4 main.py \
--model suit_base_224 \
--batch-size 256 \
--epochs 300 \
--num_workers 32 \
--spix-method fastslic \
--data-set IMNET \
--data-path datasets/imagenet-1k \
--output_dir outputs/suit_base \
--lr 8e-4 \
--weight-decay 8e-2 \
--opt adamw \
--drop-path 0.3 \
--clip-grad 1.0 \
--n-spix-segments 196 \
--downsample 2 \
--pe-type ff \
--pe-injection concat \
--aggregate max avg \
--mixup 0.0 \
--cutmix 1.0 \
--remode rand \
--trial_name suit_base