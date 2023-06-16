from typing import Optional

from torch import nn
from torchvision.models import resnet18, efficientnet_b3, resnet152, vgg19, alexnet

class TransferNetwork(nn.Module):
	def __init__(self):
		super(TransferNetwork, self).__init__()
		self.architecture = "resnet"
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