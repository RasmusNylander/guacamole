import os 
os.chdir('guacamole')
import json
import torch
from metrics import IoU
from dataloader import TACO,DatasetType

iou_threshold   = 0.5
resize_to       = 600

root_dir=os.getcwd()
anns_file_path = os.path.join(root_dir, 'annotations.json')
with open(anns_file_path, 'r') as f:
	dataset = json.loads(f.read())
annotations = dataset['annotations']
ann_bb      = torch.tensor([ann['bbox'] for ann in annotations])
ann_cat     = torch.tensor([ann['category_id'] for ann in annotations])
ann_img_id  = torch.tensor([ann['image_id'] for ann in annotations])

image_info  = dataset['images']
img_hw      = torch.tensor([[img['height'],img['width']] for img in image_info])

bb_file_path = os.path.join(root_dir, "bounding_boxes_fast.pt")
proposals = torch.load(bb_file_path)


for img_id,prop_bb in enumerate(proposals):

    true_bb = ann_bb[ann_img_id==img_id]
    cat_bb  = ann_cat[ann_img_id==img_id]

    x_fraction = resize_to / img_hw[img_id,0]
    y_fraction = resize_to / img_hw[img_id,1]
    scaler = torch.tensor([x_fraction, y_fraction]*2)
    true_bb = true_bb * scaler


    iou_bb = IoU(prop_bb,true_bb)

    prop_cat = cat_bb[iou_bb.argmax(1)]
    prop_cat[iou_bb.max(1)[0]<0.5] = 60

