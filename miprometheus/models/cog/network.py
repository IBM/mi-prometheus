"""
Defines the COG model
"""

__author__ = "Emre Sevgen"

import torch
import numpy as np
import torch.nn as nn
from miprometheus.models.cog.vstm import VSTM
from miprometheus.models.cog.ops import FeatureAttention, SpatialAttention, SemanticAttention
from miprometheus.models.model import Model

# Model inherits from Module, which is very useful.
# But for now, inherit directly from Module for testing
class CogModel(Model):

	# Temporary default value for params
	def __init__(self, params, problem_default_values_={}):
		#params = ParamInterface()
		super(CogModel,self).__init__(params,problem_default_values_)

		self.name = 'CogModel'

		self.data_definitions = {'images': {'size': [-1,-1,-1,-1,-1], 'type': [torch.Tensor]},
														'questions': {'size': [-1,-1,-1], 'type': [list,str]},
														'targets_class': {'size': [-1,-1,-1], 'type': [torch.Tensor]},
														'targets_reg': {'size' : [-1,-1,2], 'type': [torch.Tensor]}
														}

		# Initialize word lookup dictionary
		self.word_lookup = {}

		# Initialize unique word counter. Updated by UpdateAndFetchLookup
		self.nr_unique_words = 0

		# This should be the length of the longest sentence encounterable
		self.nwords = 24
		self.words_embed_length = 64
		self.vocabulary_size = 256
		self.nr_classes = 2

		self.controller_input_size = self.nwords + 5*5*128 + 5*5*3
		self.controller_output_size = 128

		self.lstm_input_size = self.words_embed_length
		self.lstm_hidden_units = 64

		self.vstm_shape=(5,5)
		self.vstm_inchannels = 128
		self.vstm_outchannels = 3
		self.vstm_nmaps = 4
		self.vstm_controlinputsize = self.controller_output_size*2

		self.VisualProcessing()
		self.SemanticProcessing()
		self.EmbedVocabulary(self.vocabulary_size,self.words_embed_length)
		self.Controller(self.controller_output_size)
		self.VisualMemory(self.vstm_shape,
											self.vstm_inchannels,
											self.vstm_outchannels,
											self.vstm_nmaps,
											self.vstm_controlinputsize)

		self.feature_attn1 = FeatureAttention(64,self.controller_output_size*2)
		self.feature_attn2 = FeatureAttention(128,self.controller_output_size*2)
		self.spatial_attn1 = SpatialAttention(128,self.controller_output_size*2)
		self.semantic_attn1 = SemanticAttention(self.lstm_hidden_units*2,self.controller_output_size*2)

		self.classifier1 = nn.Linear(128,self.nr_classes)
		self.pointer1 = nn.Linear(5*5*3,49)

	def forward(self, data_dict):
		images = data_dict['images'].permute(1,0,2,3,4)
		questions = data_dict['questions']
		#targets_class = data_dict['targets_class']
		#targets_reg = data_dict['targets_reg']
		questions = self.forward_lookup2embed(questions)
		
		output_class = torch.zeros((images.size()[1],images.size()[0],2),requires_grad=False)
		output_point = torch.zeros((images.size()[1],images.size()[0],49),requires_grad=False)
		attention = torch.randn((images.size()[1],self.controller_output_size*2),requires_grad=False)
		controller_state = torch.zeros((1,images.size()[1],128),requires_grad=False)
		vstm_state = torch.zeros((images.size()[1],self.vstm_nmaps,self.vstm_shape[0],self.vstm_shape[1]),requires_grad=False)

		for j, image_seq in enumerate(images):
			classification, pointing, attention, vstm_state, controller_state = self.forward_full_oneseq(
			image_seq,questions, attention, vstm_state,controller_state)

			output_class[:,j:j+1,:] = classification[:,0:1,:]
			output_point[:,j:j+1,:] = pointing[:,0:1,:]

		return output_class, output_point
		

	def forward_full_oneseq(self,
							images,
							questions,
							attention,
							vstm_state,
							controller_state):

		#print('Attention size: {}'.format(attention.size()))
		#print('Controller_State size: {}'.format(controller_state.size()))
		
		out_cnn1 = self.forward_img2cnn_attention(images,attention)
	
		#print('questions: {}'.format(questions))
		#questions = self.forward_lookup2embed(questions)
		out_lstm1, state_lstm1 = self.forward_embed2lstm(questions)
		out_semantic_attn1 = self.semantic_attn1(out_lstm1,attention)
		#print('out_semantic_attn1 size: {}'.format(out_semantic_attn1.size()))
		out_vstm1, vstm_state = self.vstm1(out_cnn1,vstm_state,attention)

		in_controller1 = torch.cat((out_semantic_attn1.view(-1,1,self.nwords),out_cnn1.view(-1,1,128*5*5),out_vstm1.view(-1,1,3*5*5)),-1)
		out_controller1, controller_state = self.controller1(in_controller1,controller_state)

		classification = self.classifier1(out_controller1.view(-1,1,128))
		pointing = self.pointer1(out_vstm1.view(-1,1,5*5*3))
		attention = torch.cat((out_controller1.squeeze(),controller_state.squeeze()),-1)
		return classification, pointing, attention, vstm_state, controller_state
		

	def forward_img2cnn(self,images):

		out_conv1 		= self.conv1(images)
		out_maxpool1	= self.maxpool1(out_conv1)
		out_conv2			= self.conv2(out_maxpool1)
		out_maxpool2	= self.maxpool2(out_conv2)
		out_conv3 		= self.conv3(out_maxpool2)
		out_maxpool3	= self.maxpool3(out_conv3)
		out_conv4			= self.conv4(out_maxpool3)
		out_maxpool4	= self.maxpool4(out_conv4)

		return out_maxpool4

	def forward_img2cnn_attention(self,images,attention):
		out_conv1 		= self.conv1(images)
		out_maxpool1	= self.maxpool1(out_conv1)
		out_conv2			= self.conv2(out_maxpool1)
		out_maxpool2	= self.maxpool2(out_conv2)
		out_conv3 		= self.conv3(out_maxpool2)
		out_maxpool3	= self.maxpool3(out_conv3)
		out_feature_attn1, attn_feature_attn1  = self.feature_attn1(out_maxpool3,attention)
		out_conv4 = self.conv4(out_feature_attn1)
		out_maxpool4 = self.maxpool4(out_conv4)
		out_feature_attn2, attn_feature_attn2 = self.feature_attn2(out_maxpool4,attention)
		out_spatial_attn1, attn_spatial_attn1 = self.spatial_attn1(out_feature_attn2,attention)

		return out_spatial_attn1		

	# For a single timepoint in a single sample, returns (nwords,64)
	def forward_words2embed(self,questions):
		
		out_embed=torch.zeros(len(questions),self.nwords,self.words_embed_length)
		for i, sentence in enumerate(questions):
			for j, word in enumerate(sentence[0].split()):
				#print('j is {} and word is {}'.format(j,word))
				out_embed[i,j,:] = ( self.Embedding(self.UpdateAndFetchLookup(word)) )
		
		return out_embed

	def forward_lookup2embed(self,questions):
		
		out_embed=torch.zeros(questions.size(0),self.nwords,self.words_embed_length)
		for i, sentence in enumerate(questions):
			out_embed[i,:,:] = ( self.Embedding( sentence ))
		
		return out_embed

	# For a single timepoint in a single sample, returns (nwords,128)
	def forward_embed2lstm(self,out_embed):
	
		out_lstm1, (c_n,h_n) = self.lstm1(out_embed)
		return out_lstm1, (c_n,h_n)

	# Visual Processing
	# Currently lacking attention on the last two layers
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

	#print(repr(lstm))
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

	# Semantic Processing
	# For a single timepoint in a single sample, returns (nwords,128)
	def SemanticProcessing(self):
		# 128 unit Bidirectional LSTM
		# torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

		# Input is a 64-dim embedding.
		# Hidden state is 128 - this is what they are referring to as 128 unit.
		# Then I assume it's a single layer.
		# Output is 64x2 = 128
		self.lstm1 = nn.LSTM(self.lstm_input_size,self.lstm_hidden_units,1, batch_first=True,bidirectional=True)

	#Controller Unitd
	def Controller(self,nr_units):
		# Undefined (!) number of GRU units. Not sure if Bidirecitional.
		# torch.nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

		# Input is a concatenation of:
			# Post-attention activity of top visual layer through a 128-unit fully connected layer. (Size = 128)
			# Semantic memory. (Size = nword ? )
			# vSTM Module output. 
		# "In addition, the activity of the top visual layer is summed up across space and provided to the controller." (??)
		self.controller1 = nn.GRU(self.controller_input_size, nr_units,batch_first=True)

	def VisualMemory(self,shape,in_channels,out_channels,n_maps,control_input_size):
		self.vstm1 = VSTM(shape,in_channels,out_channels,n_maps,control_input_size)

	# Embed vocabulary for all available task families
	# COG paper used a 64-dim training vector.
	# For a single timepoint in a single sample, returns (nwords,64)
	def EmbedVocabulary(self,vocabulary_size,words_embed_length):
		self.Embedding = nn.Embedding(vocabulary_size,words_embed_length)

	# Given a single word, updates lookup table if necessary, then returns embedded vector
	def UpdateAndFetchLookup(self,word):
		if word not in self.word_lookup:
			self.word_lookup[word] = self.nr_unique_words
			self.nr_unique_words += 1
		return torch.tensor([self.word_lookup[word]], dtype=torch.long)

	def EmbedQuestions(self,questions):
		return self.forward_words2embed(questions)
	

if __name__ == '__main__':
	from miprometheus.utils.param_interface import ParamInterface
	from miprometheus.problems.seq_to_seq.vqa.cog import COG
	import os
	import torch.optim as optim

	params = ParamInterface()
	tasks = ['AndCompareColor']
	params.add_config_params({'data_folder': os.path.expanduser('~/data/cog'), 'root_folder': ' ', 'set': 'val', 'dataset_type': 'canonical','tasks': tasks})

	# Create problem - task Go
	cog_dataset = COG(params)

	# Set up Dataloader iterator
	from torch.utils.data import DataLoader
	
	dataloader = DataLoader(dataset=cog_dataset, collate_fn=cog_dataset.collate_fn,
		            batch_size=64, shuffle=False, num_workers=8)

	# Get a batch64))
	#print(repr(lstm))
	batch = next(iter(dataloader))

	# Initialize model
	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	#params.add_config_params(params_dict)
	model = CogModel(params)

	images = batch['images'].permute(1,0,2,3,4)
	questions = batch['questions']
	#embedded_questions = model.EmbedQuestions(questions)
	#print(embedded_questions.size())

	# Test forward pass of image64))
	#print(repr(lstm))
	#print(model.forward_img2cnn(images[0]).size())

	# Test forward pass of words
	#embed = model.forward_words2embed(batch['questions'][0][0].split())
	#print(repr(embed))
	#lstm = model.forward_embed2lstm(embed.view(1,128,64))
	#print(repr(lstm))

	# Test full forward pass
	out_pass1, pointing, attention,vstm_state,controller_state = model.forward_full_oneseq(images[0],questions)
	#print(out_pass1)
	#print(out_pass1.size())
	#exit(0)
	out_pass2 = model.forward_full_oneseq(images[1],questions,attention,vstm_state,controller_state)

	
	# Try training!
	# Switch to sequence-major representation to make this much easier.	

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)

	from tensorboardX import SummaryWriter
	writer = SummaryWriter('runs/exp1/')
	#print(images[0].size())
	#print(embedded_questions.size())

	#dummy_images = torch.autograd.Variable(torch.Tensor(64,3,112,112),requires_grad=True)
	#dummy_questions = torch.autograd.Variable(torch.Tensor(64,32,64),requires_grad=True)
	#dummy_out_class,dummy_out_reg,_,_,_ = model.forward_full_oneseq(dummy_images,dummy_questions)
	#writer.add_graph(CogModel().forward_full_oneseq,(dummy_images,dummy_questions))

	for epoch in range(2):
	
		running_loss = 0.0
		for i, data in enumerate(dataloader,0):

			optimizer.zero_grad()
			targets = data['targets_class']
		
			for j, target in enumerate(targets):
				targets[j] = [0 if item=='false' else 1 for item in target]
			targets = torch.LongTensor(targets)
		
			output = model(data)
	
			loss = criterion(output.view(-1,2),targets.view(64*4))
			loss.backward()
			optimizer.step()

			writer.add_scalar('loss',loss.item(),i)
			#writer.add_graph('model',model,(images[0],questions))

			#running_loss += loss.item()
			print(i)
			print('Loss: {}'.format(loss.item()))






	
	
	
