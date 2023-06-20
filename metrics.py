import torch
import torchvision
from torch import Tensor
import torchmetrics

@torch.jit.script
def IoU(rect1: Tensor, rect2: Tensor) -> Tensor:
	rect1[:, 2:] = rect1[:, :2] + rect1[:, 2:]
	rect2[:, 2:] = rect2[:, :2] + rect2[:, 2:]
	return torchvision.ops.box_iou(rect1, rect2)


@torch.jit.script
def cross_entropy(prediction_logits: Tensor, target: Tensor) -> Tensor:
	return torch.nn.functional.cross_entropy(prediction_logits, target)


def get_metrics() -> list[torchmetrics.Metric]:
	# precision, recall, f1, accuracy, iou
	return [
		torchmetrics.Precision(task="multiclass", num_classes=60, average="macro"),
		torchmetrics.Recall(task="multiclass", num_classes=60, average="macro"),
		torchmetrics.F1Score(task="multiclass", num_classes=60, average="macro"),
		torchmetrics.Accuracy(task="multiclass", num_classes=60, average="macro"),
		torchmetrics.JaccardIndex(task="multiclass", num_classes=60, average="macro")
	]


if __name__ == "__main__":
	get_metrics()