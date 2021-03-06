import yaml

yaml_path = r"./models/yolov3-spp.yaml"

with open(yaml_path, 'r') as f:
    yaml_file = yaml.load(f, yaml.FullLoader)

ch = yaml_file['ch'] = yaml_file.get('ch', 3)

anchors = yaml_file['anchors']
print(type(anchors))

import torch.nn as nn


print(eval('nn.Conv2d')(3, 3, 1, 1))

