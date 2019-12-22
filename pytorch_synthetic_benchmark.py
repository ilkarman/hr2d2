from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import datetime
import torchvision
from torchvision import transforms
from pprint import pprint


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Synthetic Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 32)",
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist-url", type=str, help="url used to set up distributed training"
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument("--rank", default=-1, type=int, help="rank of the worker")
parser.add_argument("--ngpus_per_node", default=-1, type=int, help="ngpus_per_node")
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)
parser.add_argument(
    "--sriov", action="store_true", default=False, help="uses sriov",
)
parser.add_argument("--model", type=str, default="resnet50", help="model to benchmark")
parser.add_argument(
    "--num-warmup-batches",
    type=int,
    default=10,
    help="number of warm-up batches that don't count towards benchmark",
)
parser.add_argument(
    "--num-batches-per-iter",
    type=int,
    default=10,
    help="number of batches per benchmark iteration",
)
parser.add_argument(
    "--num-iters", type=int, default=10, help="number of benchmark iterations"
)

parser.add_argument(
    "--num-accumulations",
    type=int,
    default=1,
    help="number of batches to locally accumulate",
)

parser.add_argument(
    "--dataset", type=str, default="synthetic", help="help"
)  # synthetic, fake


def main():
    pprint(os.environ)
    print(os.getenv("NCCL_SOCKET_IFNAME"))
    print(os.getenv("NCCL_DEBUG"))
    print(os.getenv("NCCL_DEBUG_SUBSYS"))
    print(os.getenv("NCCL_IB_DISABLE"))
    args = parser.parse_args()
    print(f"Distributed Backend: {args.dist_backend}")
    print(f"SRIOV: {args.sriov}")
    print(f"Datetime: {datetime.datetime.now().isoformat()}")
    print(f"Dataset: {args.dataset}")
    args.distributed = args.world_size >= 2 or args.ngpus_per_node > 1
    print(args)
    if args.distributed:
        print("Running distributed")
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        print("Running single GPU")
        main_worker(0, args)


def main_worker(local_gpu_id, args):
    
    print(f"Original device {torch.cuda.current_device()}")
    print(f"Device count {torch.cuda.device_count()}")
    orig_gpu = local_gpu_id
    # local_gpu_id = 1
    print("Use GPU: {} for training".format(local_gpu_id))
    torch.cuda.set_device(local_gpu_id)
    torch.cuda.empty_cache()
    print(f"Set device {torch.cuda.current_device()}")
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = (args.rank * args.ngpus_per_node) + orig_gpu
    # device_ids = list(range(args.rank * args.ngpus_per_node, (args.rank + 1) * args.ngpus_per_node))
    print(f"Total rank {args.rank}")
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url if args.dist_url else "env://",
            world_size=args.world_size,
            rank=args.rank,
        )

    # cudnn.benchmark = True
   
    # Set up standard model.
    model = getattr(models, args.model)()
    model.cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_gpu_id], find_unused_parameters=True
        )

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Set up fixed fake data
    data = torch.randn(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    faked_data = torchvision.datasets.FakeData(
        size=(args.num_warmup_batches * args.batch_size * args.world_size)
        + (
            args.num_iters
            * args.world_size
            * args.batch_size
            * args.num_accumulations
            * args.num_batches_per_iter
        ),
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=transforms.ToTensor(),
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            faked_data, num_replicas=args.world_size, rank=args.rank
        )
    else:
        sampler = torch.utils.data.sampler.RandomSampler(faked_data)

    fake_loader = iter(
        torch.utils.data.DataLoader(
            faked_data,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
        )
    )

    def fake_benchmark_step():
        data, target = next(fake_loader)
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        for i in range(args.num_accumulations):
            output = model(data)
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def fake_accumulate_benchmark_step():
        optimizer.zero_grad()
        with model.no_sync():
            for i in range(args.num_accumulations - 1):
                data, target = next(fake_loader)
                data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
        data, target = next(fake_loader)
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def accumulate_benchmark_step():
        optimizer.zero_grad()
        with model.no_sync():
            for i in range(args.num_accumulations - 1):
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def benchmark_step():
        optimizer.zero_grad()
        for i in range(args.num_accumulations):
            output = model(data)
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    def log(s, nl=True):
        if args.rank != 0:
            return
        print(s, end="\n" if nl else "")

    if args.distributed:
        if "fake" in args.dataset:
            step = fake_accumulate_benchmark_step
        else:
            step = accumulate_benchmark_step
    else:
        if "fake" in args.dataset:
            step = fake_benchmark_step
        else:
            step = benchmark_step

    log("Model: %s" % args.model)
    log("Batch size: %d" % args.batch_size)
    device = "GPU"
    log("Number of %ss: %d" % (device, args.world_size))

    # Warm-up
    log("Running warmup...")
    timeit.timeit(step, number=args.num_warmup_batches)

    # Benchmark
    log("Running benchmark...")
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(step, number=args.num_batches_per_iter)
        img_sec = (
            args.batch_size * args.num_accumulations * args.num_batches_per_iter / time
        )
        log("Iter #%d: %.1f img/sec per %s" % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log("Img/sec per %s: %.1f +-%.1f" % (device, img_sec_mean, img_sec_conf))
    log(
        "Total img/sec on %d %s(s): %.1f +-%.1f"
        % (
            args.world_size,
            device,
            args.world_size * img_sec_mean,
            args.world_size * img_sec_conf,
        )
    )


if __name__ == "__main__":
    main()
