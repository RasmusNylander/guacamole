import torch
from torch import Tensor
import cv2
from tqdm import tqdm

from dataloader import TACO


def extract_proposals(dataset: TACO, fast: bool=True) -> list[Tensor]:
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

	bounding_boxes = []
	for image, _, _ in tqdm(dataset, unit="image", desc="Extracting proposals"):
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
	bounding_boxes = extract_proposals(dataset)

	print("Saving bounding boxes")
	torch.save(bounding_boxes, "bounding_boxes.pt")
	exit(0)
