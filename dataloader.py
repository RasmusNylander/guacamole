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


	def __init__(self, root_dir: PathLike="/dtu/datasets1/02514/data_wastedetection", ds_type: DatasetType = DatasetType.all, data_augmentation= None):
		self.root_dir = root_dir
		self.type = ds_type
		self.data_augmentation = data_augmentation
		self.tacoitems = {}

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
		return len(self.tacoitems)

	def __getitem__(self, index):
		tacoitem = self.tacoitems[self.img_ids[index]]
		image = torchvision.io.read_image(tacoitem.path)
		return image, tacoitem.bboxs, tacoitem.categories



if __name__ == '__main__':
	dataset = TACO()
	id = 0
	print(dataset[id])


