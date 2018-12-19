#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model
from miprometheus.models.mental_model.memory import Memory
from miprometheus.models.mental_model.ops import SemanticAttention


class MentalModel(Model):
	def __init__(self,params):
		super(MentalModel,self).__init__(params)

		# Define parameters
		# Visual processing
		self.in_channels = 3
		self.layer_channels = [24,24,24,24]
		self.features_shape = 112
		for channel in self.layer_channels:
			self.features_shape = np.floor((self.features_shape-2)/2)
		self.features_shape = int(self.features_shape)

		# Semantic processing
		self.vocabulary_size = 256
		self.words_embed_length = 64
		self.lstm_input = 64
		self.lstm_hidden = 64
		self.nwords = 16

		# Controller
		self.controller_input = 2*(self.layer_channels[3] + 3) + 128
		self.controller_hidden = 256
		self.pondering = 3

		# Memory
		self.batch_size = 48
		self.memory_size = 8
		self.object_size = self.layer_channels[3] + 3


		# Define the layers
		# Define visual processing layers
		self.conv1 = nn.Conv2d(self.in_channels,self.layer_channels[0],3)
		self.maxpool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(self.layer_channels[0],self.layer_channels[1],3)
		self.maxpool2 = nn.MaxPool2d(2)
		self.conv3 = nn.Conv2d(self.layer_channels[1],self.layer_channels[2],3)
		self.maxpool3 = nn.MaxPool2d(2)
		self.conv4 = nn.Conv2d(self.layer_channels[2],self.layer_channels[3],3)
		self.maxpool4 = nn.MaxPool2d(2)

		# Define semantic processing layers
		self.embedding = nn.Embedding(self.vocabulary_size,self.words_embed_length,padding_idx=0)
		self.lstm1 = nn.LSTM(self.lstm_input,self.lstm_hidden,bidirectional=True)

		# Define the controller
		self.controller1 = nn.GRU(self.controller_input,self.controller_hidden)

		# Define working memory
		self.memory = Memory(self.batch_size,self.memory_size,self.object_size,self.controller_hidden)
		self.attention_init = nn.Parameter(torch.randn((1,1, self.controller_hidden))) 

		# Define Bahdanau attention
		self.semantic_attn1 = SemanticAttention(self.lstm_input*2,self.controller_hidden)
	


	def forward(self,images,question):

		y = torch.zeros((questions.size(0),self.nwords,self.words_embed_length))
		for i, sentence in enumerate(questions):
			y[i,:,:] = self.embedding(sentence)
		y, _ = self.lstm1(y)
		
		for l, image in enumerate(images):
			x = self.conv1(image)
			x = nn.functional.relu(self.maxpool1(x))
			x = self.conv2(x)
			x = nn.functional.relu(self.maxpool2(x))
			x = self.conv3(x)
			x = nn.functional.relu(self.maxpool3(x))
			x = self.conv4(x)
			x = nn.functional.relu(self.maxpool4(x))

			controller_out = self.attention_init.expand(1,self.batch_size,-1)
			for i in range(self.features_shape):
				for j in range(self.features_shape):
					obj = torch.cat((x[:,:,i,j], 
													torch.Tensor([[i,j,l]]).expand(self.batch_size,-1)),dim=-1)
					mem_obj = torch.zeros_like(obj)
					for k in range(self.pondering):
						z = self.semantic_attn1(y,controller_out.squeeze())
						controller_in = torch.cat((obj,mem_obj,z),dim=-1)
						controller_out, _ = self.controller1(controller_in.unsqueeze(0))
	
						read_key = self.memory.read_keygen(controller_out.squeeze())
						read_subset = torch.sigmoid(self.memory.read_subset_gen(controller_out.squeeze()))
						read_content_address = self.memory.subset_similarity(read_key,read_subset)
						read_location_address = torch.nn.functional.softmax(self.memory.read_location(controller_out.squeeze()),dim=-1)
						read_address_mix = torch.nn.functional.softmax(self.memory.read_mix_gen(controller_out.squeeze()),dim=-1)
						read_address = self.memory.address_mix(read_content_address,read_location_address,read_address_mix)
						read_gate = torch.sigmoid(self.memory.read_gate(controller_out.squeeze()))
						mem_obj = self.memory.read(read_address,read_gate)

						erase_key = self.memory.erase_keygen(controller_out.squeeze())
						erase_subset = torch.sigmoid(self.memory.erase_subset_gen(controller_out.squeeze()))
						erase_content_address = self.memory.subset_similarity(erase_key,erase_subset)
						erase_location_address = torch.nn.functional.softmax(self.memory.erase_location(controller_out.squeeze()),dim=-1)
						erase_address_mix = torch.nn.functional.softmax(self.memory.erase_mix_gen(controller_out.squeeze()),dim=-1)
						erase_address = self.memory.address_mix(erase_content_address,erase_location_address,erase_address_mix)
						erase_gate = torch.sigmoid(self.memory.erase_gate(controller_out.squeeze()))
						self.memory.erase(erase_address,erase_gate)

						write_key = self.memory.write_keygen(controller_out.squeeze())
						write_subset = torch.sigmoid(self.memory.write_subset_gen(controller_out.squeeze()))
						write_content_address = self.memory.subset_similarity(write_key,write_subset)
						write_location_address = torch.nn.functional.softmax(self.memory.write_location(controller_out.squeeze()),dim=-1)
						write_address_mix = torch.nn.functional.softmax(self.memory.write_mix_gen(controller_out.squeeze()),dim=-1)
						write_address = self.memory.address_mix(write_content_address,write_location_address,write_address_mix)
						write_gate = torch.sigmoid(self.memory.write_gate(controller_out.squeeze()))
						self.memory.write(write_address,write_gate,obj)
						
					
		
if __name__ == '__main__':
	# Simulate Cog for now.
	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	mm = MentalModel(params)

	# images = [sequence x batch x channels x width x height]
	images = torch.rand((4,2,3,112,112))

	# questions = [batch x sequence of ints]
	questions = torch.randint(0,10,(2,16),dtype=torch.long)

	mm(images,questions)
	print(mm.memory.memory)
	

	

		




		
