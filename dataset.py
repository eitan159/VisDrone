from mmyolo.registry import DATASETS
from mmyolo.datasets import YOLOv5CocoDataset
import os.path as osp
from mmengine.dataset import BaseDataset

@DATASETS.register_module()
class VisDrone(YOLOv5CocoDataset):
    pass
    # def parse_data_info(self, raw_data_info):
    #     data_info = raw_data_info
    #     data_split = self.data_prefix["img"] 
    #     data_info['img_path'] = osp.join(data_split ,data_info['img_path'])
    #     return data_info