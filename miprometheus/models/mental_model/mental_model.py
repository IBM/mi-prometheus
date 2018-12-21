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

# Notes:

# 1) Is mixing between content-based and location based addressing good? Do we just get garbage read/write to two locations instead?
# 2) Can we initialize memory to a non-homogeneous state? This could improve content-based addressing at the start of a sequence.
# 3) Is Sharpening required?
# 4) Are normalizations, activation functions, and range of values reasonable across the network?
# 5) The CNN is likely highly suboptimal. Improve?
# 6) Right now the LSTM processing the questions is a bit weird.
#			The fetching network expects the last output of the LSTM to encode relevant information about all objects required (for fetching at least)
#			However, given pondering and attention mechanism on controller, the LSTM is also expected to 'word by word' encode meaning?
#			This might be fine, as fetching network also gets controller output?
#			However, this is the same as the relational networks, which have for inputs two objects and the last output of the LSTM.
#			In the RL paper, this seemed to be enough for questions in CLEVR, right? - Double check.
# 7) Training occasionally explodes with nans ?!?
# 		Made batch size flexible for forward(..), maybe that fixes it.
#			This means the self.batch_size parameter is kinda the maximum batch size.
#			Might still blow up. Perhaps memory should be resized for each batch.
#			Now memory is also reset to batch size. Hopefully now its fixed.
#			Sometimes the loss explodes with weird gradients - maybe its training parameters?
#			Weirdly, restarting from recent parameters doesnt lead to explosion...
#			Did not fix. Is the error in problem set or model?
#			Restarting does eventually lead to explosion, but not instantly.
# 8) Currently, we augment object candidates with normalized x,y coordinates (-1 to 1) and normalized time (0 to 1)
#			This is somewhat inspired from RN paper. Would be curious to see whether x, y information is used for pointing
#			In a similar vein, would be interesting to see if time information is utilized for memory / relational tasks
#			Actually, to make the claim that this is approximating a Mental Model, need to show that object encoding does successfully capture relational qualities
#			In this framework, how would a nonsensical answer appear? Low confidence? Would this need to be trained as its own class?
#			In the classical example of competing mental models (A is left of B, C is left of B, is A left of C?) what would we expect to see?
#			Unfortunately, this is hard to test with COG, as we have explicit coordinates.
# 9) In RN, sentences are object candidates. This feels too coarse. Can/should words be object candidates? 
#			Perhaps this was practical, as a loop over all objects would be too insane with words.
#			That is not a problem with current setup, as we do not loop over objects, just pick out two.
# 10)Related: we have a hard limit of two objects compared. We also don't have a zero-object comparison.
#			Technically, non relational (single-object) questions should be fine by fetching the same object twice.
#			Would be interesting to see if this is indeed the case.



class MentalModel(Model):
	def __init__(self,params,problem_default_values_={}):
		super(MentalModel,self).__init__(params,problem_default_values_)

		# Set default params
		params.add_default_params({'num_classes' : problem_default_values_['num_classes'],
															 'vocab_size' : problem_default_values_['embed_vocab_size']})

		# Get app state gpu/cpu
		self.dtype = self.app_state.dtype

		# Define parameters
		# Visual processing
		self.in_channels = 3
		self.layer_channels = [24,24,24,24]
		self.features_shape = 112
		for channel in self.layer_channels:
			self.features_shape = np.floor((self.features_shape-2)/2)
		self.features_shape = int(self.features_shape)
		# Normalization
		self.img_norm = 255.0

		# Semantic processing
		self.vocabulary_size = params['vocab_size']
		self.words_embed_length = 64
		self.lstm_input = 64
		self.lstm_hidden = 64
		self.nwords = 24

		# Controller
		self.controller_input = 2*(self.layer_channels[3] + 3) + 128
		self.controller_hidden = 256
		self.pondering = 3

		# Memory
		self.memory_size = 8
		self.object_size = self.layer_channels[3] + 3

		# Output
		self.output_classes_class=params['num_classes']
		self.output_classes_point=49

		# Misc
		self.sequence_length = 4




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
		self.lstm1 = nn.LSTM(self.lstm_input,self.lstm_hidden,bidirectional=True,batch_first=True)
		self.lstm_cell_init= nn.Parameter(torch.randn((2,1,self.lstm_hidden),requires_grad=True).type(self.dtype)*0.1)
		self.lstm_hidden_init= nn.Parameter(torch.randn((2,1,self.lstm_hidden),requires_grad=True).type(self.dtype)*0.1)

		# Define the controller
		self.controller1 = nn.GRU(self.controller_input,self.controller_hidden,batch_first=True)
		self.controller_init = nn.Parameter(torch.randn((1,1,self.controller_hidden),requires_grad=True).type(self.dtype)*0.1)

		# Define working memory
		self.memory = Memory(self.memory_size,self.object_size,self.controller_hidden,self.app_state)
		self.attention_init = nn.Parameter(torch.randn((1,1, self.controller_hidden),requires_grad=True).type(self.dtype)*0.1) 

		# Define Bahdanau attention
		self.semantic_attn1 = SemanticAttention(self.lstm_input*2,self.controller_hidden)

		# Define object fetch network
		# Input controller state, controller hidden state, question LSTM final state
		# Output two concatenated read vectors
		self.objectfetch = nn.Linear(2*self.controller_hidden+self.lstm_input*2,2*self.controller_hidden)

		# Define relational networks
		self.relationalnet_class = nn.Linear(2*self.object_size + self.lstm_input*2,self.output_classes_class)
		self.relationalnet_point = nn.Linear(2*self.object_size + self.lstm_input*2,self.output_classes_point)




		# Initialize all networks
		# CNN
		nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu'))

		self.conv1.bias.data.fill_(0.01)
		self.conv2.bias.data.fill_(0.01)
		self.conv3.bias.data.fill_(0.01)
		self.conv4.bias.data.fill_(0.01)

		# Semantic
		for name, param in self.lstm1.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param,0.01)
			elif 'weight' in name:
				nn.init.xavier_uniform_(param)
		
		# Controller
		for name, param in self.controller1.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param,0.01)
			elif 'weight' in name:
				nn.init.xavier_uniform_(param)

		# Fetch network
		nn.init.xavier_uniform_(self.objectfetch.weight)
		self.objectfetch.bias.data.fill_(0.01)	

		# Output
		nn.init.xavier_uniform_(self.relationalnet_class.weight)
		self.relationalnet_class.bias.data.fill_(0.01)
		
		nn.init.xavier_uniform_(self.relationalnet_point.weight)
		self.relationalnet_point.bias.data.fill_(0.01)
		

	def forward(self,data_dict):

		images = data_dict['images'].permute(1,0,2,3,4) / self.img_norm
		questions = data_dict['questions']

		if(self.check_and_print_nan(images)):
			print('images')

		self.memory.reset(images.size(1))

		y = torch.zeros((questions.size(0),self.nwords,self.words_embed_length))
		for i, sentence in enumerate(questions):
			y[i,:,:] = self.embedding(sentence)
		y, _ = self.lstm1(y,( self.lstm_hidden_init.expand(-1,images.size(1),-1).contiguous(),
												  self.lstm_cell_init.expand(-1,images.size(1),-1).contiguous() ) )

		if(self.check_and_print_nan(y)):
			print('y')
		
		output_class = torch.zeros((images.size(1),images.size(0),self.output_classes_class),requires_grad=False).type(self.dtype)
		output_point = torch.zeros((images.size(1),images.size(0),self.output_classes_point),requires_grad=False).type(self.dtype)

		for l, image in enumerate(images):
			x = self.conv1(image)
			x = nn.functional.relu(self.maxpool1(x))
			x = self.conv2(x)
			x = nn.functional.relu(self.maxpool2(x))
			x = self.conv3(x)
			x = nn.functional.relu(self.maxpool3(x))
			x = self.conv4(x)
			x = nn.functional.relu(self.maxpool4(x))
			
			controller_out = self.attention_init.expand(image.size(0),1,-1)
			controller_hidden = self.controller_init.expand(1,image.size(0),-1)
			for i in range(self.features_shape):
				for j in range(self.features_shape):
					obj = torch.cat((x[:,:,i,j], 
													(torch.Tensor([[i-0.5*(self.features_shape-1),j-0.5*(self.features_shape-1),l]])/torch.Tensor([[0.5*(self.features_shape-1),0.5*(self.features_shape-1),self.sequence_length-1]])).expand(image.size(0),-1)),dim=-1)
					mem_obj = torch.zeros_like(obj)


					if(self.check_and_print_nan(obj)):
						print('obj')
					if(self.check_and_print_nan(mem_obj)):
						print('mem_obj')



					for k in range(self.pondering):
						z = self.semantic_attn1(y,controller_out.squeeze())
						controller_in = torch.cat((obj,mem_obj,z),dim=-1)
						controller_out, controller_hidden = self.controller1(controller_in.unsqueeze(1),controller_hidden)

						if(self.check_and_print_nan(controller_in)):
							print('controller_in')
						if(self.check_and_print_nan(controller_out)):
							print('controller_out')
						
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
						
			concatenated_reads = self.objectfetch(torch.cat((controller_out.squeeze(),controller_hidden.squeeze(),y[:,-1,:]),dim=-1))
			read1, read2 = torch.split(concatenated_reads,(self.controller_hidden,self.controller_hidden),-1)

			read_key = self.memory.read_keygen(read1.squeeze())
			read_subset = torch.sigmoid(self.memory.read_subset_gen(read1.squeeze()))
			read_content_address = self.memory.subset_similarity(read_key,read_subset)
			read_location_address = torch.nn.functional.softmax(self.memory.read_location(read1.squeeze()),dim=-1)
			read_address_mix = torch.nn.functional.softmax(self.memory.read_mix_gen(read1.squeeze()),dim=-1)
			read_address = self.memory.address_mix(read_content_address,read_location_address,read_address_mix)
			read_gate = torch.sigmoid(self.memory.read_gate(read1.squeeze()))
			mem_obj1 = self.memory.read(read_address,read_gate)	
		
			read_key = self.memory.read_keygen(read2.squeeze())
			read_subset = torch.sigmoid(self.memory.read_subset_gen(read2.squeeze()))
			read_content_address = self.memory.subset_similarity(read_key,read_subset)
			read_location_address = torch.nn.functional.softmax(self.memory.read_location(read2.squeeze()),dim=-1)
			read_address_mix = torch.nn.functional.softmax(self.memory.read_mix_gen(read2.squeeze()),dim=-1)
			read_address = self.memory.address_mix(read_content_address,read_location_address,read_address_mix)
			read_gate = torch.sigmoid(self.memory.read_gate(read2.squeeze()))
			mem_obj2 = self.memory.read(read_address,read_gate)	
			
			output_class[:,l,:] = self.relationalnet_class(torch.cat((mem_obj1,mem_obj2,y[:,-1,:]),dim=-1))
			output_point[:,l,:] = self.relationalnet_point(torch.cat((mem_obj1,mem_obj2,y[:,-1,:]),dim=-1))
		
		return output_class, output_point

	def check_and_print_nan(self,tensor):
		if torch.isnan(tensor).any():
			print(tensor)
			return True
		else:
			return False
					
		
if __name__ == '__main__':
	# Simulate Cog for now.
	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	mm = MentalModel(params)

	# images = [sequence x batch x channels x width x height]
	images = torch.rand((4,48,3,112,112))

	# questions = [batch x sequence of ints]
	questions = torch.randint(0,10,(48,16),dtype=torch.long)

	output = mm(images,questions)
	print(mm.memory.memory)
	print(output)
	

	

		




		
