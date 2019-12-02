import logging
import logging.config
import os
import yacs
import fire
import torch
from default import _C as config
from models.cls_hrnet_3d import get_cls_net as get_cls_net_3d
from models.cls_hrnet_2dplus1 import get_cls_net as get_cls_net_2dplus1
from models.cls_hrnet_2dplus1_se import get_cls_net as get_cls_net_2dplus1_cse
from models.r2plus1d import R2Plus1DClassifier


def run_r2plus1d_orig(*options):
    # ORIGINAL R2+1D
    # Equiv to 34 layer
    model = R2Plus1DClassifier(num_classes=400, layer_sizes=[3,4,6,3])

    model.cuda()
    model.train()

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((2, 3, 16, 112, 112)).cuda()
    output = model(input)  # Run one pass

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time

    # Run 100 batches
    for i in range(100):
        start_time = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        if i >= 10:
            avg_time += (time.time() - start_time)
            avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Number of Parameters : {}".format(count_parameters(model)))
    print("Average Running Time: {}s".format(avg_time/(100)))
    print("Average GPU Memory: {}MiB".format(avg_mem/(100*1024)))
    # 34 Layer [3,4,6,3]
    #Number of Parameters : 63735310
    #Average Running Time: 0.06696544885635376s
    #Average GPU Memory: 6426875.9MiB
    # 18 Layer [2,2,2,2]
    #Number of Parameters : 33396910
    #Average Running Time: 0.03813744068145752s
    #Average GPU Memory: 4060387.58MiB

def run_3d(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_3d(config)

    model.cuda()
    model.train()

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((2, 3, 16, 112, 112)).cuda()
    output = model(input)  # Run one pass

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time

    # Run 100 batches
    for i in range(100):
        start_time = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        if i >= 10:
            avg_time += (time.time() - start_time)
            avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Number of Parameters : {}".format(count_parameters(model)))
    print("Average Running Time: {}s".format(avg_time/(100)))
    print("Average GPU Memory: {}MiB".format(avg_mem/(100*1024)))
    #Number of Parameters : 51063308
    #Average Running Time: 0.12405349969863892s
    #Average GPU Memory: 4214270.64MiB


def run_2dplus1(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    # Quiker to run but takes up for GPU vRAM

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_2dplus1(config)
    #print(model)
    model.cuda()
    model.train()

    # Can do a max batch of 2 if 16 frames and 112*112

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((2, 3, 16, 112, 112)).cuda()
    output = model(input)  # Run one pass

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time

    # Run 100 batches
    for i in range(100):
        start_time = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        if i >= 10:
            avg_time += (time.time() - start_time)
            avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Number of Parameters : {}".format(count_parameters(model)))
    print("Average Running Time: {}s".format(avg_time/(100)))
    print("Average GPU Memory: {}MiB".format(avg_mem/(100*1024)))
    #Number of Parameters : 51112991
    #Average Running Time: 0.13220649242401122s
    #Average GPU Memory: 12514048.465MiB

def run_2dplus1_cse(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    # Quiker to run but takes up for GPU vRAM

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_2dplus1_cse(config)
    #print(model)
    model.cuda()
    model.train()

    # Can do a max batch of 2 if 16 frames and 112*112

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((2, 3, 16, 112, 112)).cuda()
    output = model(input)  # Run one pass

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    avg_time = 0
    avg_mem  = 0
    import time

    # Run 100 batches
    for i in range(100):
        start_time = time.time()
        outputs = model(input)
        torch.cuda.synchronize()
        if i >= 10:
            avg_time += (time.time() - start_time)
            avg_mem += (torch.cuda.max_memory_allocated() - mem)

    print("Number of Parameters : {}".format(count_parameters(model)))
    print("Average Running Time: {}s".format(avg_time/(100)))
    print("Average GPU Memory: {}MiB".format(avg_mem/(100*1024)))
    #Number of Parameters : 51152095
    #Average Running Time: 0.14000601053237915s
    #Average GPU Memory: 12749557.595MiB

if __name__ == "__main__":
    #fire.Fire(run_r2plus1d_orig)
    #fire.Fire(run_3d)
    #fire.Fire(run_2dplus1)
    fire.Fire(run_2dplus1)
