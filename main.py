from dataset import VisDrone
from mmengine.config import Config
from mmengine.runner import Runner
import argparse

def dynamic_cfg(args):
    cfg.work_dir = args.work_dir
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.train_cfg.max_epochs = args.epochs

    # wandb
    cfg.visualizer.vis_backends[0]['init_kwargs']['project'] = args.wandb_project


    # data path fixes
    cfg.train_dataloader.dataset.data_root = args.data_root
    cfg.val_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_root = args.data_root


parser = argparse.ArgumentParser()

parser.add_argument("--data_root", 
                    type=str,
                    required=True, 
                    help="path for your data splits directories")

parser.add_argument("--work_dir", 
                    type=str,
                    default='./exp/', 
                    help="where to save the ckpt and logs")

parser.add_argument("--wandb_project", 
                    type=str,
                    default='VisDrone', 
                    help="wandb project name")

# model settings
parser.add_argument("--model_size", 
                    type=str,
                    default="s", 
                    choices="s, x",
                    help="which PP-YOLOE size to train")

parser.add_argument("--lr", 
                    type=float,
                    default=0.001, 
                    help="lr")

parser.add_argument("--epochs", 
                    type=int,
                    default=30, 
                    help="epochs")

args = parser.parse_args()

if args.model_size == "s":
    cfg_path =  './configs/ppyoloe_plus_s_fast_8xb8-80e_coco.py'
elif args.model_size == "x":
    cfg_path =  './configs/ppyoloe_plus_x_fast_8xb8-80e_coco.py'
else:
    raise Exception("I didn't implemented more versions of PPYOLOE+")

cfg = Config.fromfile(cfg_path)
dynamic_cfg(args)

runner = Runner.from_cfg(cfg)
runner.train()
runner.test()
