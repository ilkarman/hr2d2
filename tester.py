import logging
import logging.config
import os
import yacs
import fire
import torch
from default import _C as config
from models.cls_hrnet_3d import get_cls_net as get_cls_net_3d
from models.cls_hrnet_2d import get_cls_net as get_cls_net_2d
from models.cls_hrnet_2dplus1 import get_cls_net as get_cls_net_2dplus1



def run_3d(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_3d(config)

    model.cuda()
    model.train()

    # Can do a max batch of 6 if 16 frames and 112*112
    # nvidia-smi tops at 15.8GB

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((6, 3, 16, 112, 112)).cuda()
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
    #Number of Parameters : 53753956
    #Average Running Time: 0.192874276638031s
    #Average GPU Memory: 12591148.05MiB


# BIGGER???
def run_2dplus1(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_2dplus1(config)
    print(model)
    model.cuda()
    model.train()

    # Can do a max batch of 2 if 16 frames and 112*112
    # ???? more params

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
    #Number of Parameters : 53801753
    #Average Running Time: 0.1259375762939453s
    #Average GPU Memory: 11663663.045MiB



def run_2d(*options, 
    cfg='/home/iliauk/notebooks/hr2d2/configs/cls_hrnet_w18.yaml'):

    config.defrost()
    config.merge_from_file(cfg)

    model = get_cls_net_2d(config)

    model.cuda()
    model.train()

    # nvidia-smi tops at 9.9GB

    mem = torch.cuda.max_memory_allocated()
    input = torch.randn((6, 3, 224, 224)).cuda()
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
    #Number of Parameters : 21262012
    #Average Running Time: 0.06875052213668824s
    #Average GPU Memory: 7981788.57MiB


if __name__ == "__main__":
    #fire.Fire(run_3d)
    fire.Fire(run_2dplus1)
    #fire.Fire(run_2d)