import torch
import torchvision
from torch import Tensor

@torch.jit.script
def IoU(rect1: Tensor, rect2: Tensor) -> Tensor:
	rect1[:, 2:] = rect1[:, :2] + rect1[:, 2:]
	rect2[:, 2:] = rect2[:, :2] + rect2[:, 2:]
	return torchvision.ops.box_iou(rect1, rect2)

