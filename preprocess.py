import os
import os.path as osp
import json
from PIL import Image

def ann_to_dict(ann, ann_id, image_id):
    ann = ann.split(",")
    bbox = [int(ann[0]), int(ann[1]), int(ann[2]), int(ann[3])] # x, y, w, h
    area = bbox[2] * bbox[3] # w * h
    return {"bbox": bbox, "category_id": int(ann[5]), "area": area, "id": ann_id,
            "image_id": image_id, "iscrowd": 0}


def get_anns(ann_path, original_idx2class, new_class2idx):
    with open(ann_path, "r") as f:
        anns = f.readlines()
    
    instances = []
    for ann in anns:
        ann = ann_to_dict(ann)
        ann_class = original_idx2class[ann["category_id"]]
        if ann_class not in new_class2idx:
            continue
        ann["category_id"] = new_class2idx[ann_class] 
        instances.append(ann)
    
    return instances


def create_json_for_visdrone(root_dir, split):
    ann_dir = f"{root_dir}/{split}/annotations/"
    imgs_dir = f"{root_dir}/{split}/images/"

    original_idx2class = {0: "ignored_regions", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van", 
                        6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others"}
    
    new_class2idx = {"pedestrian": 0, "people": 1, "bicycle": 2, "car": 3, "van": 4, 
                     "truck": 5, "bus":6 , "motor": 7}

    categories = [{"id": idx, "name": name} for name, idx in new_class2idx.items()]

    images = []
    annotations = []
    ann_id = 0
    for i, (ann_path, img_path) in enumerate(zip(os.listdir(ann_dir), os.listdir(imgs_dir))):
        assert ann_path.split(".")[0] == img_path.split(".")[0]
        img_sample = {}
        image_id = i
        img = Image.open(osp.join(imgs_dir, img_path))
        img_sample["height"] = img.height
        img_sample["width"] = img.width
        img_sample["file_name"] = img_path
        img_sample["id"] = image_id
        images.append(img_sample)
        
        with open(osp.join(ann_dir, ann_path), "r") as f:
            anns = f.readlines()
    
        for ann in anns:
            ann = ann_to_dict(ann, ann_id, image_id)
            ann_class = original_idx2class[ann["category_id"]]
            if ann_class not in new_class2idx:
                continue
            ann["category_id"] = new_class2idx[ann_class]
            ann_id += 1
            annotations.append(ann)
            
    
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)

    json.dump(coco_format_json, open(f"visdrone_{split}.json", "w"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", 
                    type=str,
                    default="", 
                    help="path to visdrone data files")
    args = parser.parse_args()

    create_json_for_visdrone(args.data_root, "train")
    create_json_for_visdrone(args.data_root, "val")
    create_json_for_visdrone(args.data_root, "test")


