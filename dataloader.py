import os
from enum import Enum
from os import PathLike
from typing import Optional, Type

import torch
import torchvision.transforms.functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import json

import warnings
warnings.filterwarnings('ignore')


class DatasetType(Enum):
	train = 1
	test = 2
	validation = 3

class TACOItem():
	def __init__(self, path: str, bboxs: Tensor = None, categories: Tensor = None):              
		self.path = path
		self.bboxs = bboxs
		self.categories = categories

class TACO(torch.utils.data.Dataset):
	rand = np.arange(1500) #TODO check if all indexes present

	TEST_INDICES = rand[-250:]
	VALIDATION_INDICES = rand[:250]
	TRAINING_INDICES = rand[250:1250]


	def __init__(self, root_dir: PathLike="/dtu/datasets1/02514/data_wastedetection", ds_type: DatasetType = DatasetType.train, data_augmentation= None):
		self.root_dir = root_dir
		self.type = ds_type
		self.data_augmentation = data_augmentation
		self.tacoitems = {}

		anns_file_path = os.path.join(root_dir, 'annotations.json')
		with open(anns_file_path, 'r') as f:
			dataset = json.loads(f.read())

		imgs = dataset['images']
		for img in imgs:
			id = img['id']
			path = os.path.join(root_dir, img['file_name'])
			tacoitem = TACOItem(path)
			self.tacoitems[id] = tacoitem

		annotations = dataset['annotations']

		for annotation in annotations:
			id = annotation['image_id']
			tacoitem = self.tacoitems[id]

			bbox = torch.tensor([[float(x) for x in annotation['bbox']]])
			if tacoitem.bboxs is None:
				tacoitem.bboxs = bbox
				tacoitem.categories = torch.tensor([int(annotation['category_id'])])
			else:
				tacoitem.bboxs = torch.cat([tacoitem.bboxs, bbox], dim=0)
				tacoitem.categories= torch.cat([tacoitem.categories, 
												torch.tensor([int(annotation['category_id'])])], dim=0)

		categories = dataset['categories']
		
		assert len(DatasetType) == 3, f"Unexpected number of DatasetTypes in {self.__class__.__name__}.__init__! Expected 3, got {len(DatasetType)}"
		if DatasetType.train == type:
			indices_mask = TACO.TRAINING_INDICES
		elif DatasetType.test == type:
			indices_mask = TACO.TEST_INDICES
		else:
			indices_mask = TACO.VALIDATION_INDICES

	def __len__(self) -> int:
		return len(self.tacoitems)

	def resize(self, image, bboxs, resize_to = 600):

		resized_image = torchvision.transforms.functional.resize(image, size = (resize_to, resize_to))

		x_fraction = resize_to / image.shape[2]
		y_fraction = resize_to / image.shape[1]

		scaler = torch.tensor([x_fraction, y_fraction]*2)

		resized_bboxs = bboxs * scaler
		return resized_image, resized_bboxs

	def __getitem__(self, id):
		tacoitem = self.tacoitems[id]
		image = torchvision.io.read_image(tacoitem.path)
		reized_image, resized_bboxs = self.resize(image, tacoitem.bboxs) 
		return reized_image, resized_bboxs, tacoitem.categories



if __name__ == '__main__':
	dataset = TACO()
	id = 1
	image, bboxs, cats = dataset[id]

	import matplotlib.pyplot as plt
	from matplotlib.patches import Polygon, Rectangle
	fig,ax = plt.subplots(1)
	plt.imshow(image.permute(1, 2, 0).numpy())
	x, y, w, h = bboxs[0].numpy()
	rect = Rectangle((x,y),w,h,linewidth=2, facecolor='none', edgecolor='orange')
	ax.add_patch(rect)
	plt.savefig("try_resized.png")


