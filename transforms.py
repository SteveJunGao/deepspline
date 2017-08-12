import torch
import random
import numpy as np

class Compose(object):
	"""
	Compose several augmentation together
	"""
	def __init__(self, transforms):
		self.transforms = transforms
	def __call__(self, img, targets = None):
		for t in self.transforms:
			img, targets = t(img, targets)
		return img, targets

class ToTensor(object):
	"""
	Conver a image (HxWxC) to Torch Tensor (CxHxW)
	Also scale it from (0,255) to (0,1.0)
	Will not Change the target.
	"""
	def __call__(self, img, target=None):
		img = torch.from_numpy(img)
		img = img.permute(2,0,1)
		img = img / 255
		return img, target


class RandomVerticalFlip(object):
	"""
	Flip an image vertically
	"""
	def __init__(self, p):
		self.p = p
	def __call__(self, img, target):
		img = img[:, ::-1, :].contiguous()


class RandomHorizonFlip(object):
	"""
	Flip an image horizontally
	"""
	def __init__(self, p):
		self.p = p
	def __call__(self, img, target):
		if random.random() < self.p:
			img = img[:, ::-1]
