from models.cls_hrnet_2dplus1_frn import get_cls_net
import torch
from default import _C as config
from default import update_config

# def test_model(capsys):
#     update_config(config, config_file='/data/home/mat/repos/hr2d2/configs/cls_hrnet_w18_moments_frn.yaml')
#     data = torch.ones((2, 3, 32, 128, 128)).cuda()
#     model = get_cls_net(config).cuda()
#     output = model(data)
#     assert output.shape == (2, 39) 
