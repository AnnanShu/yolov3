import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py'
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *

try:
    import thop # for FLOPS computation
except ImportError:
    thop = None

class Model(nn.Module):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, yaml.FullLoader)

        # define model
        ch = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml.get('nc'):
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']}g with nc={nc}")
            self.yaml['nc'] = nc  # overwrite yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), [ch])


def parse_model(d, ch):  # config dict, in-channel
    logger.info('')
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, save-list, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval string

    return d, ch