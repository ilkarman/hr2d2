import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def test_save_model_ddp():
    """Save DDP wrapped model and load as DDP wrapped model

    If you save a DDP wrapped model with
    
        torch.save(model.state_dict(), "test.pth")
    
    Then when you want to load a saved model
    you need to instanciate the model, then wrap it in DDP and finally load it

        model = ToyModel().cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
        model.load_state_dict(torch.load("test.pth"))
    Otherwise if you try to load it without it wrapped in DDP it will not load correctly
    """
    local_process_id = 0
    dist_url = "env://"
    dist_backend = "nccl"
    world_size = 1
    local_gpu_id = local_process_id
    rank = local_process_id

    torch.cuda.set_device(local_gpu_id)
    torch.distributed.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank,
    )
    device = "cuda"
    data = torch.ones((10)).cuda()
    model = ToyModel().cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True
    )
    output = model(data)
    torch.save(model.state_dict(), "test.pth")
    dist.barrier()
    del model
    torch.cuda.empty_cache()
    model = ToyModel().cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True
    )
    output2 = model(data)
    model.load_state_dict(torch.load("test.pth"))
    output3 = model(data)
    assert torch.all(torch.eq(output, output3))
    assert not torch.all(torch.eq(output, output2))
    dist.destroy_process_group()


def test_save_model():
    """Save DDP wrapped model and load as single GPU/CPU model

    If you save a DDP wrapped model you need to use the .module in 
    order for it to be loadable on a single GPU/CPU
    
        torch.save(model.module.state_dict(), "test.pth")
    
    Then when you want to load a saved model
    you can simply instanciate the model and load it

        model = ToyModel().cuda()
        model.load_state_dict(torch.load("test.pth"))
    """
    local_process_id = 0
    dist_url = "env://"
    dist_backend = "nccl"
    world_size = 1
    local_gpu_id = local_process_id
    rank = local_process_id
    torch.cuda.set_device(local_gpu_id)
    torch.distributed.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank,
    )
    device = "cuda"
    data = torch.ones((10)).cuda()
    model = ToyModel().cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True
    )
    output = model(data)
    torch.save(model.module.state_dict(), "test.pth")
    dist.barrier()
    del model
    torch.cuda.empty_cache()
    model = ToyModel().cuda()
    output2 = model(data)
    model.load_state_dict(torch.load("test.pth"))
    output3 = model(data)
    assert torch.all(torch.eq(output, output3))
    assert not torch.all(torch.eq(output, output2))
    dist.destroy_process_group()

