import argparse
import json
import os
from ADP.inpainting_metric import get_inpainting_metrics

args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('--real_path', default=None, type=str)
args.add_argument('--fake_path', default=None, type=str)
args.add_argument('--info_txt', default=None, type=str)
args = args.parse_args()
real_path = os.path.join(args.real_path)
fake_path = os.path.join(args.fake_path)
info_txt = os.path.join(args.info_txt)
metric = get_inpainting_metrics(real_path, fake_path)
print(metric)
with open(info_txt, "w") as f:
    f.write(json.dumps(metric))