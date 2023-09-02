from dataset import VisDrone
from mmengine.config import Config
from mmengine.runner import Runner
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_size", 
                    type=str,
                    default="s", 
                    choices="s, x",
                    help="which PP-YOLOE size to train")

args = parser.parse_args()

if args.model_size == "s":
    cfg_path =  './configs/ppyoloe_plus_s_fast_8xb8-80e_coco.py'
elif args.model_size == "x":
    cfg_path =  './configs/ppyoloe_plus_x_fast_8xb8-80e_coco.py'
else:
    raise Exception("I didn't implemented more versions of PPYOLOE+")

cfg = Config.fromfile(cfg_path)
runner = Runner.from_cfg(cfg)
runner.train()
runner.test()
