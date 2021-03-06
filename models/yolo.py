import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py'
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import MixConv2d, CrossConv

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

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x if x < 0 else x + 1] for x in f])
        elif m is Detections:
            pass

    return 1, 2