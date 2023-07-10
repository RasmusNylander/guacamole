import os
import json
import torch
from metrics import IoU
from dataloader import TACO,DatasetType

iou_threshold   = 0.5
resize_to       = 600
DATA_ROOT= r"D:\data"
PROPOSAL_DIR = r".\proposals"

anns_file_path = os.path.join(DATA_ROOT, 'annotations.json')
with open(anns_file_path, 'r') as f:
	dataset_metainfo = json.loads(f.read())
annotations = dataset_metainfo['annotations']
annotated_bounding_boxes = torch.tensor([ann['bbox'] for ann in annotations])
bounding_boxes_categories = torch.tensor([ann['category_id'] for ann in annotations])
bounding_box_image_ids = torch.tensor([ann['image_id'] for ann in annotations])

image_info  = dataset_metainfo['images']
image_hw      = torch.tensor([[img['height'], img['width']] for img in image_info])

bb_file_path = os.path.join(PROPOSAL_DIR, "bounding_boxes_fast.pt")
proposals = torch.load(bb_file_path)


print(len(torch.cat(proposals)))
proposals_cat = []
num_obj = 0
obj_found = 0
true_boxes = []
for image_id, image_proposals in enumerate(proposals):

	image_bounding_boxes = annotated_bounding_boxes[bounding_box_image_ids == image_id]
	image_bounding_boxes_categories = bounding_boxes_categories[bounding_box_image_ids == image_id]
	num_obj += len(image_bounding_boxes_categories)

	x_fraction = resize_to / image_hw[image_id,0]
	y_fraction = resize_to / image_hw[image_id,1]
	scaler = torch.tensor([x_fraction, y_fraction]*2)
	image_bounding_boxes = torch.round(image_bounding_boxes * scaler)

	true_boxes.append(image_bounding_boxes)
	# proposals[img_id] = torch.cat([proposals[img_id],true_bb])

	iou_bb = IoU(image_proposals, image_bounding_boxes)

	prop_cat = iou_bb.argmax(1) #prop_cat = cat_bb[iou_bb.argmax(1)]
	prop_cat[iou_bb.max(1)[0]<iou_threshold] = 60
	proposals_cat.append(prop_cat)
	obj_found += (len(torch.unique(prop_cat))-1)

print(len(torch.cat(proposals_cat)))
print(len(torch.cat(proposals)))

# torch.save(proposals, f"bounding_boxes_quality_X.pt")
# torch.save(proposals_cat, f"bounding_boxes_qual_categories_X.pt")

