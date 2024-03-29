
import os
from enum import Enum
from os import PathLike
from typing import Optional, Type, Union

import torch
import torchvision.transforms.functional
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image, ExifTags

import sys

import warnings
warnings.filterwarnings('ignore')


class DatasetType(Enum):
	all = 0
	train = 1
	test = 2
	valid = 3


class TACOItem():
	def __init__(self, path: str, bboxs: Tensor = None, categories: Tensor = None):              
		self.path = path
		self.bboxs = bboxs
		self.categories = categories

class TACO(torch.utils.data.Dataset):

	LABELS = ['Aluminium foil', 'Battery', 'Aluminium blister pack', 'Carded blister pack', 'Other plastic bottle', 'Clear plastic bottle', 'Glass bottle', 'Plastic bottle cap', 'Metal bottle cap', 'Broken glass', 'Food Can', 'Aerosol', 'Drink can', 'Toilet tube', 'Other carton', 'Egg carton', 'Drink carton', 'Corrugated carton', 'Meal carton', 'Pizza box', 'Paper cup', 'Disposable plastic cup', 'Foam cup', 'Glass cup', 'Other plastic cup', 'Food waste', 'Glass jar', 'Plastic lid', 'Metal lid', 'Other plastic', 'Magazine paper', 'Tissues', 'Wrapping paper', 'Normal paper', 'Paper bag', 'Plastified paper bag', 'Plastic film', 'Six pack rings', 'Garbage bag', 'Other plastic wrapper', 'Single-use carrier bag', 'Polypropylene bag', 'Crisp packet', 'Spread tub', 'Tupperware', 'Disposable food container', 'Foam food container', 'Other plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Plastic straw', 'Paper straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette', 'Background']
	LABEL_SUPER_CATEGORY = {'Aluminium foil': 'Aluminium foil', 'Battery': 'Battery', 'Aluminium blister pack': 'Blister pack', 'Carded blister pack': 'Blister pack', 'Other plastic bottle': 'Bottle', 'Clear plastic bottle': 'Bottle', 'Glass bottle': 'Bottle', 'Plastic bottle cap': 'Bottle cap', 'Metal bottle cap': 'Bottle cap', 'Broken glass': 'Broken glass', 'Food Can': 'Can', 'Aerosol': 'Can', 'Drink can': 'Can', 'Toilet tube': 'Carton', 'Other carton': 'Carton', 'Egg carton': 'Carton', 'Drink carton': 'Carton', 'Corrugated carton': 'Carton', 'Meal carton': 'Carton', 'Pizza box': 'Carton', 'Paper cup': 'Cup', 'Disposable plastic cup': 'Cup', 'Foam cup': 'Cup', 'Glass cup': 'Cup', 'Other plastic cup': 'Cup', 'Food waste': 'Food waste', 'Glass jar': 'Glass jar', 'Plastic lid': 'Lid', 'Metal lid': 'Lid', 'Other plastic': 'Other plastic', 'Magazine paper': 'Paper', 'Tissues': 'Paper', 'Wrapping paper': 'Paper', 'Normal paper': 'Paper', 'Paper bag': 'Paper bag', 'Plastified paper bag': 'Paper bag', 'Plastic film': 'Plastic bag & wrapper', 'Six pack rings': 'Plastic bag & wrapper', 'Garbage bag': 'Plastic bag & wrapper', 'Other plastic wrapper': 'Plastic bag & wrapper', 'Single-use carrier bag': 'Plastic bag & wrapper', 'Polypropylene bag': 'Plastic bag & wrapper', 'Crisp packet': 'Plastic bag & wrapper', 'Spread tub': 'Plastic container', 'Tupperware': 'Plastic container', 'Disposable food container': 'Plastic container', 'Foam food container': 'Plastic container', 'Other plastic container': 'Plastic container', 'Plastic glooves': 'Plastic glooves', 'Plastic utensils': 'Plastic utensils', 'Pop tab': 'Pop tab', 'Rope & strings': 'Rope & strings', 'Scrap metal': 'Scrap metal', 'Shoe': 'Shoe', 'Squeezable tube': 'Squeezable tube', 'Plastic straw': 'Straw', 'Paper straw': 'Straw', 'Styrofoam piece': 'Styrofoam piece', 'Unlabeled litter': 'Unlabeled litter', 'Cigarette': 'Cigarette', 'Background': 'Background'}

	ALL_INDICES = np.arange(0,1500)
	TEST_INDICES = [ 324,  901,  209,  334,  979,  583,  856,  617,  582,   64, 1356,
		929,  490,  923,  158,  172,  252,  242,  136, 1134,  858, 1476,
		779,   82,  442,  467, 1393,  821,  628,  471, 1070,   96,  808,
		576, 1063, 1204,  190,  654,  990,  873,  447, 1337, 1121,  650,
		429,  377,  847,  914,   35, 1141, 1191,  223, 1082,  974,  778,
		882,  802,    0, 1276,  600, 1091,  811,  506,  695,  596,  750,
		 91,  557,   61, 1419,  747, 1239,  189,  733,  145,  941,  433,
	   1221,  965,  744,  194,  829,  299, 1408, 1467,  498,  946,  989,
		409, 1015,  562,  736, 1318, 1098,  662, 1062, 1255,   36,  541,
	   1011,   72,  296, 1114,  260, 1101,  799, 1446,  414, 1415,  922,
	   1232,  426,  551,  315,  753, 1375,  138, 1143,  590,   60, 1441,
		678,  151,  147,  961, 1447, 1041,  881,  646,  428, 1320, 1340,
		125,   10, 1172,  323, 1376, 1083, 1492, 1412,  109,  943,  588,
	   1345,  597, 1262,   67, 1139,  373,  363, 1426,  973, 1248,    8,
		 93,  322,  717, 1087,  188,  559,  772,   74,  231,  735,  274,
	   1043,  525,  972, 1004,  431,  207,  713,  665,  674, 1017,  868,
		497,  959,  317, 1344,  764,  916,  443, 1443,  304,  620,  719,
		 68, 1105, 1216,  968,  215, 1177,  456, 1293,  469,  484,  977,
	   1313,  649,  527, 1396,  167,   16, 1325, 1129,  268,  639,  226,
	   1061,  571,  342,  441,   58,  723,  918,  708,  287,  403,  143,
	   1451, 1358, 1026, 1301, 1474,  202, 1179,  144, 1400,   13,  743,
	   1173, 1431,  797, 1365,  104,  401,   77,  405, 1079, 1405,  626,
		948, 1334,  819,  413,  419,  464,  102,  131]
	VALIDATION_INDICES = [ 246, 1387,  281,  521,  810,  731, 1290,  121,  542,  910, 1421,
	   1190,   59,  463,  511,  448,  768,  884, 1450, 1028, 1067,  383,
	   1210, 1410,   52, 1213,  182,  142,  297, 1119,  230, 1484, 1364,
	   1033,  917, 1185,  243, 1457, 1268, 1057, 1430,  545, 1169, 1353,
		 50,   37, 1471, 1214,  642,  107,  589, 1460,  742,  475,  635,
	   1089,  177, 1167,  127,  726, 1429,  526,  643, 1273, 1281,  179,
		677,  466,  804,  513, 1456,  869,  417,  957,   49,  560, 1175,
		593,  384, 1110,  684,  915,  293, 1142,  830,  930, 1059, 1244,
		616,   19,  669,  962,  809,  801,  245,  235,  754,  424, 1347,
	   1138, 1453,  813,  123, 1406,  561,  340,  307,  602,  494,  552,
		937,  290, 1383,  823, 1350, 1060,  543,  416, 1224,   21, 1473,
		412, 1442,  615, 1047,    6, 1197,  889, 1225,  523,  728, 1034,
		758,  183, 1241, 1123,  955,   57,  685, 1231,  263,  885,  887,
	   1261,  911,  967,   28,  482, 1135,  704,  465, 1095,  608,  422,
		454,  985,  585,  357,  741,  656,  316,  640,  762, 1428, 1499,
	   1303,  224,   69,  956, 1486,  100,  919, 1458, 1354,  752, 1330,
		660,  896, 1439, 1074, 1277, 1021,  673, 1211,  902, 1274,  540,
		720,  278,  517,  300, 1192,  459,  939,   34,  221,   63,  124,
		327, 1444,  981, 1295, 1454,  612, 1285, 1314,  329, 1036, 1217,
		222,  170,  450,  980,  963,  806,  718,  376,  867,  828,  816,
	   1088,  629, 1220,  991,  681,  129,  632, 1147,  680, 1470,  794,
	   1263,  335,  614,  398,  958, 1307,  721, 1195, 1370, 1360, 1304,
		634,  157,  907,  520,  378,  839,  262, 1117]
	TRAINING_INDICES = np.setdiff1d(ALL_INDICES,TEST_INDICES+VALIDATION_INDICES)


	def __init__(self, root_dir: PathLike="/dtu/datasets1/02514/data_wastedetection", ds_type: DatasetType = DatasetType.all, data_augmentation = None, resize_to:int = 600):
		self.root_dir = root_dir
		self.type = ds_type
		self.data_augmentation = data_augmentation
		self.resize_to = resize_to
		self.tacoitems = {} # [img_id: TACOItem(), ...]

		if DatasetType.train == self.type:
			self.img_ids = TACO.TRAINING_INDICES
		elif DatasetType.test == self.type:
			self.img_ids = TACO.TEST_INDICES
		elif DatasetType.valid == self.type:
			self.img_ids = TACO.VALIDATION_INDICES
		elif DatasetType.all == self.type:
			self.img_ids = TACO.ALL_INDICES

		anns_file_path = os.path.join(root_dir, 'annotations.json')
		with open(anns_file_path, 'r') as f:
			dataset = json.loads(f.read())

		imgs = dataset['images']
		annotations = dataset['annotations']

		for img in imgs:
			img_id = img['id']
			if img_id in self.img_ids:
				path = os.path.join(root_dir, img['file_name'])
				tacoitem = TACOItem(path)
				self.tacoitems[img_id] = tacoitem

		print(f'Number of images in dataset: {len(self.tacoitems)}')
		annotations = dataset['annotations']

		for annotation in annotations:
			img_id = annotation['image_id']
			if img_id in self.img_ids:
				tacoitem = self.tacoitems[img_id]
				bbox = torch.tensor([[float(x) for x in annotation['bbox']]])
				if tacoitem.bboxs is None:
					tacoitem.bboxs = bbox
					tacoitem.categories = torch.tensor([int(annotation['category_id'])])
				else:
					tacoitem.bboxs = torch.cat([tacoitem.bboxs, bbox], dim=0)
					tacoitem.categories= torch.cat([tacoitem.categories, 
													torch.tensor([int(annotation['category_id'])])], dim=0)

		#categories = dataset['categories']
		
	def __len__(self) -> int:
		return len(self.img_ids)

	def resize(self, image: Tensor, bboxs: Tensor) -> tuple:
		"""
		image shape: [1, x, y] --> [1, self.resize_to, self.resize_to]
		bboxs shape: [1, n, 4] --> [1, n, 4] (but changed values)
		"""

		resized_image = torchvision.transforms.functional.resize(image, size = (self.resize_to, self.resize_to))

		for i in range(bboxs.shape[0]):
			if bboxs[i][2] > image.shape[2]:
				bboxs[i][2] = image.shape[2]
				print("width adjusted", sys.stderr)
			if bboxs[i][3] > image.shape[1]:
				bboxs[i][3] = image.shape[1]
				print("height adjusted", sys.stderr)

		x_fraction = self.resize_to / image.shape[2]
		y_fraction = self.resize_to / image.shape[1]

		scaler = torch.tensor([x_fraction, y_fraction]*2)

		resized_bboxs = bboxs * scaler
		return resized_image, resized_bboxs


	def __getitem__(self, id:int) -> tuple:
		tacoitem = self.tacoitems[self.img_ids[id]]
		image = read_image_and_rotate(tacoitem.path)
		reized_image, resized_bboxs = self.resize(image, tacoitem.bboxs) 
		return reized_image, resized_bboxs, tacoitem.categories,


def collate(batch: list[tuple[Tensor, Tensor, Tensor]]) -> list[list[Tensor]]:  # Must be here, otherwise it cannot be pickled
	image = [item[0] for item in batch]
	bboxs = [item[1] for item in batch]
	categories = [item[2] for item in batch]
	return [image, bboxs, categories]

@torch.jit.script
def clamp_bboxs(bboxses: Tensor, image_size: Tensor) -> Tensor:
	bbox_min_width = 6

	x, y, w, h = bboxses[:, 0], bboxses[:, 1], bboxses[:, 2], bboxses[:, 3]
	x = x.clamp(0, image_size[0] - bbox_min_width - 1)
	y = y.clamp(0, image_size[1] - bbox_min_width - 1)

	x2, y2 = x + w, y + h
	x2 = x2.clamp(x + bbox_min_width, image_size[0])
	y2 = y2.clamp(y + bbox_min_width, image_size[1])

	return torch.stack([x, y, x2, y2], dim=1)


@torch.jit.script
def bounding_box_to_coordinates(bboxses: Tensor) -> Tensor:
	x, y, w, h = bboxses[:, 0], bboxses[:, 1], bboxses[:, 2], bboxses[:, 3]
	x2, y2 = x + w, y + h
	return torch.stack([x, y, x2, y2], dim=1)


class Proposals(torch.utils.data.Dataset):
	BACKGROUND_INDEX = 60
	PROPOSALS_PER_IMAGE = 4

	def __init__(self, root_dir: PathLike = "/dtu/datasets1/02514/data_wastedetection",ds_type: DatasetType = DatasetType.train,):
		self.taco = TACO(root_dir, ds_type=ds_type)
		self.bboxs = torch.load("proposals/bounding_boxes_quality_X.pt") # both proposal and gt
		self.bboxs = [self.bboxs[index] for index in self.taco.img_ids]
		# self.cumsum_proposal_ids = np.cumsum([len(x) for x in self.bboxs])
		self.categories = torch.load("proposals/bounding_boxes_qual_categories_X.pt") # both proposal and gt
		self.categories = [self.categories[index] for index in self.taco.img_ids]

		# proper_bboxes = []
		# propers_categories = []
		# for image, category in zip(self.bboxs, self.categories):
		# 	proper_bboxes.append(image[(image[:, 2] > 2) * (image[:, 3] > 2)])
		# 	propers_categories.append(category[(image[:, 2] > 2) * (image[:, 3] > 2)])
		# self.bboxs = proper_bboxes
		# self.categories = propers_categories

	def __len__(self):
		return Proposals.PROPOSALS_PER_IMAGE*len(self.taco)

	def idx_to_image_and_proposal_id(self, idx):
		image_idx = self.cumsum(self.cumsum_proposal_ids <= idx).argmax()
		proposal_idx = idx - self.cumsum_proposal_ids[image_idx]

		return image_idx, proposal_idx

	def sample_index(self, index: int) -> tuple[Tensor, Tensor]:
		taco_index = index // Proposals.PROPOSALS_PER_IMAGE
		proposals = self.bboxs[taco_index]
		proposal_categories = self.categories[taco_index]

		if index % 4 != 0:
			mask = proposal_categories == self.BACKGROUND_INDEX
		else:
			mask = proposal_categories != self.BACKGROUND_INDEX

		proposals = proposals[mask]
		proposal_categories = proposal_categories[mask]

		proposal_index = torch.randint(0, proposals.shape[0], (1,)).item()
		proposal = proposals[proposal_index]
		category = proposal_categories[proposal_index]

		proposal = proposal.int()
		return proposal, category

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
		proposal_min_width = 6
		image_width = 600
		taco_index = idx // Proposals.PROPOSALS_PER_IMAGE

		image_path = self.taco.tacoitems[self.taco.img_ids[taco_index]].path
		image = read_image_and_rotate(image_path)
		image = torchvision.transforms.functional.resize(image, size=(image_width, image_width))

		proposal, category = self.sample_index(idx)

		coordinates = clamp_bboxs(proposal.unsqueeze(dim=0), torch.tensor([image_width, image_width]))
		x, y, x2, y2 = coordinates[0, 0], coordinates[0, 1], coordinates[0, 2], coordinates[0, 3]

		patch = image[:, y:y2, x:x2]
		patch = torchvision.transforms.functional.resize(patch, size=(224, 224))

		category = torch.nn.functional.one_hot(category, num_classes=len(TACO.LABELS))

		return patch, category


class Patches(torch.utils.data.Dataset):
	BACKGROUND_INDEX = 60

	def __init__(self, image: Tensor, proposals: Tensor):
		self.proposals = torch.squeeze(proposals)
		self.image = torch.squeeze(image)

	def __len__(self):
		return len(self.proposals)

	def __getitem__(self, idx: int) -> tuple[Tensor]:
		proposal = self.proposals[idx]

		coordinates = clamp_bboxs(proposal.unsqueeze(dim=0), torch.tensor([600, 600]))
		x, y, x2, y2 = coordinates[0, 0], coordinates[0, 1], coordinates[0, 2], coordinates[0, 3]

		patch = self.image[:, y:y2, x:x2]
		patch = torchvision.transforms.functional.resize(patch, size=(224, 224))
		return patch


class ProposalsEval(Proposals):

	def __init__(self, root_dir: PathLike = "/dtu/datasets1/02514/data_wastedetection",ds_type: DatasetType = DatasetType.train):
		super(ProposalsEval, self).__init__(root_dir= root_dir,ds_type = ds_type)

		self.bboxs = torch.load("proposals/bounding_boxes_quality.pt") 

	def __len__(self):
		return len(self.taco.img_ids) 

	def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
		image, true_bboxs, true_cats = self.taco[idx]
		proposals = self.bboxs[idx]

		true_bboxs = clamp_bboxs(true_bboxs,torch.tensor([600,600]))
		true_bboxs[:,2] = true_bboxs[:,2] - true_bboxs[:,0]
		true_bboxs[:,3] = true_bboxs[:,3] - true_bboxs[:,1]

		return proposals, image, true_bboxs, true_cats


def make_dataloader(batch_size: int, dataset_path_override: Optional[PathLike], num_workers=3) -> tuple[DataLoader, DataLoader, DataLoader]:
	if dataset_path_override is not None:
		train_dataset = Proposals(dataset_path_override,  DatasetType.train)
		validation_dataset, test_dataset = Proposals(dataset_path_override, DatasetType.valid), TACO(dataset_path_override, DatasetType.test)
	else:
		train_dataset = Proposals()
		validation_dataset, test_dataset = Proposals(ds_type=DatasetType.valid), TACO(ds_type=DatasetType.test)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

	return train_loader, validation_loader, test_loader


def read_image_and_rotate(image_path):
	for orientation in ExifTags.TAGS.keys():
		if ExifTags.TAGS[orientation] == 'Orientation':
			break
		
	I = Image.open(image_path)

	# Load and process image metadata
	if I._getexif():
		exif = dict(I._getexif().items())
		# Rotate portrait and upside down images if necessary
		if orientation in exif:
			if exif[orientation] == 3:
				I = I.rotate(180,expand=True)
			if exif[orientation] == 6:
				I = I.rotate(270,expand=True)
			if exif[orientation] == 8:
				I = I.rotate(90,expand=True)

	return torchvision.transforms.functional.to_tensor(I)


