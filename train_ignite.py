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
#from models.two_stream import TwoStream

from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers import CosineAnnealingScheduler

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
    # Setup Augmentations
    # dataset.py performs random crop; maybe add random flip
   
    # Dataloaders
    n_classes = run_config.DATASET.N_CLASSES

    train_set = get_dataset(run_config.DATASET.DATASET)(
        run_config.DATASET.ROOT,
        mode=run_config.DATASET.TRAIN_SET,
        clip_len=run_config.TRAIN.CLIP_LEN,  # 32
        resize_h_w=run_config.TRAIN.RESIZE_H_W,  # (128,170)
        crop_size=run_config.TRAIN.CROP_SIZE,  # 128
        num_classes=n_classes)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = data.DataLoader(
        train_set, 
        batch_size=run_config.TRAIN.BATCH_SIZE_PER_GPU, 
        num_workers=run_config.TRAIN.NUM_WORKERS, 
        sampler=train_sampler,
    )
       
    logger.info(f"Training examples {len(train_set)}")
    logger.info(f"Train shape: {run_config.TRAIN.CROP_SIZE}")

    val_set = get_dataset(run_config.DATASET.DATASET)(
        run_config.DATASET.ROOT,
        mode=run_config.DATASET.TEST_SET,
        clip_len=run_config.TEST.CLIP_LEN,  # 32
        resize_h_w=run_config.TEST.RESIZE_H_W,  # (128,170)
        num_classes=n_classes
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

    model = get_cls_net(run_config).to(device)
    #model = TwoStream([3, 4, 6, 3]).to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True)

    # Loss and Schedule
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    # FLAG TEMP
    optimizer = optim.SGD(model.parameters(), lr=run_config.TRAIN.LR)  # hyperparameters as given in paper sec 4.1
    # 2+1 Paper: The initial learning rate is set to 0.01 and divided by 10 every 10 epochs
    # CSN Paper: Training is done in 45 epochs where we use model warming-up [14] in the first 10 epochs
    #and the remaining 35 epochs will follow the half-cosine period learning rate schedule as in [10]. The initial learning
    #rate is set to 0.01 per GPU (equivalent to 0.64 for 64 GPUs). [ 8 clips per GPU]
    # So if CSN then we need 0.08/8 = 0.01 LR
    # If 2+1 we need 0.04 = 0.025
    # Use 0.01 and do cosine annealing
    #step_scheduler = optim.lr_scheduler.StepLR(
    #    optimizer, 
    #    step_size=run_config.TRAIN.LR_STEP_SIZE, 
    #    gamma=run_config.TRAIN.LR_FACTOR  # the scheduler divides the lr by 10 every 10 epochs
    #)  
    #scheduler = LRScheduler(step_scheduler)
    snapshot_duration = 45 * len(train_loader)
    scheduler = CosineAnnealingScheduler(optimizer, "lr", config.TRAIN.LR, 0.001, snapshot_duration)

    # Setting up trainer
    trainer = create_supervised_trainer(model, optimizer, criterion, prepare_batch, device=device)

    # FLAG TEMP
    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    # Set to update the epoch parameter of our distributed data sampler so that we get
    # different shuffles
    trainer.add_event_handler(Events.EPOCH_STARTED, update_sampler_epoch(train_loader))

    if silence_other_ranks & rank != 0:
        logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    def _select_pred_and_label(model_out_dict):
        return (model_out_dict["preds"], model_out_dict["labels"])

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch,
        metrics={
            "loss": Loss(criterion, output_transform=_select_pred_and_label, device=device),
            "acc": Accuracy(output_transform=_select_pred_and_label, device=device),
        },
        device=device,
    )

    # Set the validation run to start on the epoch completion of the training run
    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluator(evaluator, val_loader))
    if rank == 0:  # Run only on master process

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, logging_handlers.log_training_output(log_interval=run_config.PRINT_FREQ),
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, logging_handlers.log_lr(optimizer))

        output_dir = generate_path(run_config.OUTPUT_DIR, git_branch(), git_hash(), run_config.MODEL.NAME, current_datetime(),)

        summary_writer = create_summary_writer(log_dir=path.join(output_dir, run_config.LOG_DIR))

        logger.info(f"Logging Tensorboard to {path.join(output_dir, run_config.LOG_DIR)}")
        
        trainer.add_event_handler(
            Events.EPOCH_STARTED, tensorboard_handlers.log_lr(summary_writer, optimizer, "epoch"),
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, tensorboard_handlers.log_training_output(summary_writer),
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            logging_handlers.log_metrics(
                "Validation results",
                metrics_dict={
                    "loss": "Avg loss :",
                    "acc": " Avg Accuracy :",
                },
            ),
        )
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            tensorboard_handlers.log_metrics(
                summary_writer,
                trainer,
                "epoch",
                metrics_dict={"loss": "Validation/Loss", "acc": "Validation/Acc",},
            ),
        )

        # FLAG: Videos need some pre-processing maybe?
        # Also would be good to visualise label and predicted label

        #evaluator.add_event_handler(
        #    Events.EPOCH_COMPLETED,
        #    tensorboard_handlers.create_video_writer(
        #        summary_writer, 
        #        "Validation/Videos",
        #        "batch"
        #    ),
        #)

        checkpoint_handler = SnapshotHandler(
            path.join(output_dir, run_config.TRAIN.MODEL_DIR),
            run_config.MODEL.NAME,
            extract_metric_from("loss"),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

        logger.info("Starting training")

    trainer.run(train_loader, max_epochs=run_config.TRAIN.END_EPOCH)


if __name__ == "__main__":
    fire.Fire(main)
