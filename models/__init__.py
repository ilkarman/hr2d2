
import models.cls_hrnet_3d
import models.cls_hrnet_2dplus1
import models.r2plus1d
import models.cls_hrnet_2dplus1_frn
from models import cls_hrnet_2dplus1_frn, cls_hrnet_2dplus1

_MODELS = {
    "3d_cls_hrnet_w18_frn": cls_hrnet_2dplus1_frn.HighResolutionNet,
    "3d_cls_hrnet_w18": cls_hrnet_2dplus1.HighResolutionNet
}

def get_cls_net(config, **kwargs):
    HighResolutionNet = _MODELS[config.MODEL.NAME]
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model