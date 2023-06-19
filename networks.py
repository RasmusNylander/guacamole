from enum import Enum
from typing import Optional

from torch import nn
from torchvision.models import resnet18, efficientnet_b3, resnet152, vgg19, alexnet

class Architecture(Enum):
	ALEXNET = "alexnet"

	def __str__(self):
		return self.value

	def from_string(s: str) -> Optional["Architecture"]:
		if s == Architecture.ALEXNET.value:
			return Architecture.ALEXNET
		else:
			return None


	def create_network(self) -> nn.Module:
		if self == Architecture.ALEXNET:
			return TransferNetwork()
		else:
			raise NotImplementedError(f"Architecture {self} not implemented")


class TransferNetwork(nn.Module):
	def __init__(self):
		super(TransferNetwork, self).__init__()
		self.architecture = Architecture.ALEXNET
		self.net = alexnet(pretrained=True)
		for param in self.net.parameters():
			param.requires_grad = False

		mod = list(self.net.classifier.children())
		mod.pop()
		mod.append(nn.Linear(4096, 60))
		new_classifier = nn.Sequential(*mod)
		self.net.classifier = new_classifier

	def forward(self, x):
		x = self.net(x)
		return x