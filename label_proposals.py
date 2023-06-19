import os 
import json
import torch
from metrics import IoU
from dataloader import TACO,DatasetType

iou_threshold   = 0.5
resize_to       = 600

os.chdir('guacamole')
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

bb_file_path = os.path.join(root_dir, "bounding_boxes_quality.pt")
proposals = torch.load(bb_file_path)


print(len(torch.cat(proposals)))
proposals_cat = []
for img_id in range(len(proposals)):

    true_bb = ann_bb[ann_img_id==img_id]
    cat_bb  = ann_cat[ann_img_id==img_id]

    x_fraction = resize_to / img_hw[img_id,0]
    y_fraction = resize_to / img_hw[img_id,1]
    scaler = torch.tensor([x_fraction, y_fraction]*2)
    true_bb = torch.round(true_bb * scaler)

    proposals[img_id] = torch.cat([proposals[img_id],true_bb])


    iou_bb = IoU(proposals[img_id],true_bb)

    prop_cat = cat_bb[iou_bb.argmax(1)]
    prop_cat[iou_bb.max(1)[0]<iou_threshold] = 60
    proposals_cat.append(prop_cat)
print(len(torch.cat(proposals_cat)))
print(len(torch.cat(proposals)))

torch.save(proposals, f"bounding_boxes_quality_X.pt")
torch.save(proposals_cat, f"bounding_boxes_qual_categories_X.pt")

