import torch
from torch import Tensor
import cv2
from tqdm import tqdm, trange

from dataloader import TACO

def extract_proposals(dataset: TACO, fast: bool=True, index: int=0) -> list[Tensor]:
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

	process_length = len(dataset) // 8
	start = index * process_length
	end = start + process_length
	print(f"Process {index} will process from {start} to {end}")

	bounding_boxes = []
	# for image, _, _ in tqdm(dataset, unit="image", desc="Extracting proposals"):
	for i in trange(start, end, unit="image", desc="Extracting proposals"):
		image, _, _ = dataset[i]
		image = image.permute(1, 2, 0).numpy()
		ss.setBaseImage(image)
		if fast:
			ss.switchToSelectiveSearchFast()
		else:
			ss.switchToSelectiveSearchQuality()

		ss.addImage(image)
		boxes = ss.process()
		bounding_boxes.append(torch.tensor(boxes))
		ss.clearImages()

	return bounding_boxes



if __name__ == '__main__':
	dataset = TACO()
	import sys
	if len(sys.argv) > 1:
		index = int(sys.argv[1])

	bounding_boxes = extract_proposals(dataset, fast=False, index=index)

	# Read index from command line



	print("Saving bounding boxes")
	torch.save(bounding_boxes, f"bounding_boxes_quality_{index}.pt")
	exit(0)
