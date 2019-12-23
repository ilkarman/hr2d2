"""Trains models using PyTorch DistributedDataParallel

To run with 8 GPUs:
MASTER_ADDR="127.0.0.1" MASTER_PORT=29500 python train_ignite.py  0 WORKERS 4 GPUS 0,1,2,3,4,5,6,7 WORLD_SIZE 8 --cfg configs/cls_hrnet_w18_moments.yaml

"""

import logging
import logging.config
import os
from os import path

import cv2
import fire
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, Normalize, Resize, PadIfNeeded
from cv_lib.utils import load_log_configuration
from cv_lib.event_handlers import (
    SnapshotHandler,
    logging_handlers,
    tensorboard_handlers,
)
from torch import nn, optim
from cv_lib.engine import (
    create_supervised_evaluator,
    create_supervised_trainer,
)

from ignite.metrics import Loss, Accuracy

from cv_lib.utils import (
    current_datetime,
    generate_path,
    git_branch,
    git_hash,
    np_to_tb,
)
from default import _C as config
from default import update_config

from ignite.engine import Events
from ignite.utils import convert_tensor
from toolz import compose, curry
from torch.utils import data
from models.cls_hrnet import get_cls_net
from ignite.contrib.handlers.param_scheduler import LRScheduler
from dataset import get_dataset
import torch.multiprocessing as mp
from cv_lib.event_handlers.logging_handlers import Evaluator
from cv_lib.event_handlers.tensorboard_handlers import create_summary_writer
from cv_lib import extract_metric_from

def prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


@curry
def update_sampler_epoch(data_loader, engine):
    data_loader.sampler.epoch = engine.state.epoch


def main(node_rank, *options, dist_url="env://", cfg=None):
    """Run training and validation of model

    Notes:
        Options can be passed in via the options argument and loaded from the cfg file
        Options from default.py will be overridden by options loaded from cfg file
        Options passed in via options argument will override option loaded from cfg file

    Args:
        node_rank(int): The rank of the node. If used on single machine it is simply 0. 
                        Across multiple machines such as with AzureML usually set to $AZ_BATCHAI_TASK_INDEX
        *options (str,int ,optional): Options used to overide what is loaded from the
                                      config. To see what options are available consult
                                      default.py
        cfg (str, optional): Location of config file to load. Defaults to None.
    """
    update_config(config, options=options, config_file=cfg)
     # Start logging
    load_log_configuration(config.LOG_CONFIG)
    print(__name__)
    logger = logging.getLogger(__name__)
    logger.debug(config.WORKERS)

    distributed = config.WORLD_SIZE >= 2
    logger.debug(config)
    if distributed:
        logger.info("Running distributed")
        mp.spawn(run, nprocs=len(config.GPUS), args=(node_rank, dist_url, config,))
    else:
        logger.info("Running single GPU")
        run(0, node_rank, dist_url, config)


def run(local_process_id, node_rank, dist_url, run_config):
    dist_backend="nccl"
    load_log_configuration(run_config.LOG_CONFIG)
    logger = logging.getLogger(__name__)
    silence_other_ranks = True
    world_size = int(run_config.WORLD_SIZE)
    distributed = world_size > 1
    local_gpu_id = run_config.GPUS[local_process_id]

    rank = (node_rank * len(run_config.GPUS)) + local_process_id

    if distributed:
        # FOR DISTRIBUTED: Set the device according to local_gpu_id.
        torch.cuda.set_device(local_gpu_id)

        # FOR DISTRIBUTED: Initialize the backend. torch.distributed.launch will
        # provide environment variables, and requires that you use init_method=`env://`.
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method=dist_url if dist_url else "env://",
            world_size=world_size,
            rank=rank,
        )

    torch.backends.cudnn.benchmark = run_config.CUDNN.BENCHMARK

    torch.manual_seed(run_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_config.SEED)
    np.random.seed(seed=run_config.SEED)
   
    # Augmentations
    val_aug = Compose(
        [
            PadIfNeeded(
                min_height=max(run_config.TEST.RESIZE_H_W),
                min_width=max(run_config.TEST.RESIZE_H_W),
                always_apply=True,
                mask_value=0,
            ),
        ]
    )

    # Dataloaders
    n_classes = run_config.DATASET.N_CLASSES

    val_set = get_dataset(run_config.DATASET.DATASET)(
        run_config.DATASET.ROOT,
        mode=run_config.DATASET.TEST_SET,
        clip_len=run_config.TEST.CLIP_LEN,  # 32
        resize_h_w=run_config.TEST.RESIZE_H_W,  # (128,170)
        crop_spatial=False,  # Don't crop 128,128 from 128,170s
        augmentations=val_aug
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank)
    val_loader = data.DataLoader(
        val_set,
        batch_size=run_config.TEST.BATCH_SIZE_PER_GPU,
        num_workers=run_config.TEST.NUM_WORKERS, 
        sampler=val_sampler,
    )

    logger.info(f"Validation examples {len(val_set)}")
    logger.info(f"Validation shape: {run_config.TEST.CROP_SIZE}")

    # Model
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        logger.warning("Can not find GPUs!!!")

 
    # Load model from file
    model = get_cls_net(run_config)
    model = model.to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True)

    model.load_state_dict(torch.load(run_config.TEST.MODEL_PATH))


    if silence_other_ranks & rank != 0:
        logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    def _select_pred_and_label(model_out_dict):
        return (model_out_dict["preds"], model_out_dict["labels"])

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch,
        metrics={
            "acc": Accuracy(output_transform=_select_pred_and_label, device=device),
        },
        device=device,
    )
    
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        logging_handlers.log_metrics(
            "Validation results",
            metrics_dict={
                "acc": " Avg Accuracy :",
            },
        ),
    )
    evaluator.run(val_loader, max_epochs=1)

if __name__ == "__main__":
    fire.Fire(main)
