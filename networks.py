import sys
from enum import Enum
from typing import Optional

from torch import nn
from torchvision.models import resnet18, efficientnet_b3, resnet152, vgg19, alexnet

from dataloader import TACO

class Architecture(Enum):
	ALEXNET = "alexnet"
	RESNET18 = "resnet18"
	RESNET152 = "resnet152"

	def __str__(self):
		return self.value

	def __repr__(self):
		if self == Architecture.ALEXNET:
			return "AlexNet"
		elif self == Architecture.RESNET18:
			return "ResNet18"
		elif self == Architecture.RESNET152:
			return "ResNet152"
		else:
			print(f"ERROR: Unknown architecture {self} in Architecture.__repr__", file=sys.stderr)
		return str(self)

	def from_string(s: str) -> Optional["Architecture"]:
		for architecture in Architecture:
			if s == str(architecture) or s == repr(architecture):
				return architecture
		return None

	def create_network(self) -> nn.Module:
		if self == Architecture.ALEXNET:
			return TransferNetwork()
		elif self == Architecture.RESNET18:
			return ResNet18()
		elif self == Architecture.RESNET152:
			return ResNet152()
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
		mod.append(nn.Linear(4096, len(TACO.LABELS)))
		new_classifier = nn.Sequential(*mod)
		self.net.classifier = new_classifier

	def forward(self, x):
		x = self.net(x)
		return x


class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()
		self.architecture = Architecture.RESNET18
		self.net = resnet18(pretrained=True)
		for param in self.net.parameters():
			param.requires_grad = False

		modelOutputFeats = self.net.fc.in_features
		self.net.fc = nn.Sequential(
			nn.Linear(modelOutputFeats, len(TACO.LABELS)),
		)

	def forward(self, x):
		x = self.net(x)
		return x


class ResNet152(nn.Module):
	def __init__(self):
		super(ResNet152, self).__init__()
		self.architecture = Architecture.RESNET152
		self.net = resnet152(pretrained=True)
		for param in self.net.parameters():
			param.requires_grad = False

		modelOutputFeats = self.net.fc.in_features
		self.net.fc = nn.Sequential(
			nn.Linear(modelOutputFeats, len(TACO.LABELS)),
		)

	def forward(self, x):
		x = self.net(x)
		return x