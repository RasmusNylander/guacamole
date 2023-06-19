import torch
import torchvision
from torch import Tensor

@torch.jit.script
def IoU(rect1: Tensor, rect2: Tensor) -> Tensor:
	rect1[:, 2:] = rect1[:, :2] + rect1[:, 2:]
	rect2[:, 2:] = rect2[:, :2] + rect2[:, 2:]
	return torchvision.ops.box_iou(rect1, rect2)


@torch.jit.script
def cross_entropy(prediction_logits: Tensor, target: Tensor) -> Tensor:
	return torch.nn.functional.cross_entropy(prediction_logits, target)