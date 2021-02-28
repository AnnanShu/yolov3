import yaml

yaml_path = "./models/yolov3-spp.yaml"

with open(yaml_path, 'r') as f:
    yaml_file = yaml.load(f, yaml.FullLoader)

ch = yaml_file['ch'] = yaml_file.get('ch', 3)

anchors = yaml_file['anchors']
type(anchors)

import torch.nn as nn


eval('nn.Conv2d')(3, 3, 1, 1)

eval