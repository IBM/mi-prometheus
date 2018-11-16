"""
Defines subunits of the COG model
"""

__author__ = "Emre Sevgen"

import torch
import numpy as np
import torch.nn as nn

#from miprometheus.models.model import Model

class COGModel:

	def __init__(self):

		self.VisualProcessing()

	def forward(self,images):

		out_conv1 		= self.conv1(images)
		out_maxpool1	= self.maxpool1(out_conv1)
		out_conv2			= self.conv2(out_maxpool1)
		out_maxpool2	= self.maxpool2(out_conv2)
		out_conv3 		= self.conv3(out_maxpool2)
		out_maxpool3	= self.maxpool3(out_conv3)
		out_conv4			= self.conv4(out_maxpool3)
		out_maxpool4	= self.maxpool2(out_conv4)

	# Visual Processing
	def VisualProcessing(self):
		# First up is a 4 layer CNN
		# Batch normalization
		# 3x3 Kernel
		# 2x2 Max Pooling after
		# ReLU

		# First Layer
		# Input to this layer is 3 channel images.
		# Output is 32 channels	
		# nn.conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
		self.conv1 = nn.Conv2d(3,32,3,stride=1,padding=0,dilation=1,groups=1,bias=True)
		# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		self.maxpool1 = nn.MaxPool2d(2,stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

		# Second Layer
		# Input to this layer is 32 channels.
		# Output is 64 channels
		# nn.conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
		self.conv2 = nn.Conv2d(32,64,3,stride=1,padding=0,dilation=1,groups=1,bias=True)
		# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		self.maxpool2 = nn.MaxPool2d(2,stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

		# Third Layer
		# Input to this layer is 64 channels.
		# Output is 64 channels
		# nn.conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
		self.conv3 = nn.Conv2d(64,64,3,stride=1,padding=0,dilation=1,groups=1,bias=True)
		# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		self.maxpool3 = nn.MaxPool2d(2,stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

		# Fourth Layer
		# Input to this layer is 64 channels.
		# Output is 128 channels
		# nn.conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True)
		self.conv4 = nn.Conv2d(64,128,3,stride=1,padding=0,dilation=1,groups=1,bias=True)
		# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		self.maxpool4 = nn.MaxPool2d(2,stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)



if __name__ == '__main__':
	from miprometheus.utils.param_interface import ParamInterface
	from miprometheus.problems.seq_to_seq.vqa.cog import COG
	import os

	params = ParamInterface()
	tasks = ['Go','CompareColor']
	params.add_config_params({'data_folder': os.path.expanduser('~/data/cog'), 'root_folder': ' ', 'set': 'val', 'dataset_type': 'canonical','tasks': tasks})

	# Create problem - task Go
	cog_dataset = COG(params)

	# Get a sample - Go
	sample = cog_dataset[0]

	# Initialize model
	model = COGModel()

	# Test forward pass
	print(model.forward(sample['images']))
	
	
