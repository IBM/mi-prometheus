"""
Defines attention subunits of the COG model
"""

__author__ = "Emre Sevgen"

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model

# Inputs: 
	# Output from conv2d layer (32,64,14,14)
	# Inputs to generate attention from (batch_size,num_units)
# Outputs: 
	# Post-attention output, same shape tensor as conv2d layer input
	# Generated attention (batch_size,attention_size == input channels)
class FeatureAttention(nn.Module):
	def __init__(self,attention_size,attention_input_size,use_mlp=False):
		super(FeatureAttention,self).__init__()
		self.attention_size = attention_size
		self.use_mlp = use_mlp
		
		self.attn1 = nn.Linear(attention_input_size,attention_size*2)

	def forward(self,inputs_to_attend,attn_gen):
	
		shift_and_scale = self.attn1(attn_gen)
		shift, scale = torch.chunk(shift_and_scale,2,-1)
		
		scale = nn.functional.relu(scale + 1.0)
	
		inputs_to_attend = (inputs_to_attend + shift.view(-1,self.attention_size,1,1)) * scale.view(-1,self.attention_size,1,1)
		inputs_to_attend = nn.functional.relu(inputs_to_attend)
		return inputs_to_attend, (shift, scale)



# Inputs:
	# Output from a conv2d layer 
	# Inputs to generate attention from (batch_size, num_units)
# Outputs:
	# Post-attention output, same shape tensor as conv2d layer input
	# Generated attention (batch_size,attention_size == input channels)
class SpatialAttention(nn.Module):
	def __init__(self,attention_size, attention_input_size):
		super(SpatialAttention,self).__init__()
		self.attention_size = attention_size
		self.attn1 = nn.Linear(attention_input_size,10)
		#self.attn1 = nn.ReLU(self.attn1)
		self.attn2 = nn.Linear(10,attention_size)
		#self.attn2 = nn.Softmax(self.attn2)
	
	def forward(self,inputs_to_attend,attn_gen):
		attn_gen = nn.functional.relu(self.attn1(attn_gen))
		attn_gen = nn.functional.softmax(self.attn2(attn_gen),dim=-1)
		inputs_to_attend *= attn_gen.view(-1,self.attention_size,1,1)

		return inputs_to_attend, attn_gen


class SemanticAttention(nn.Module):
	def __init__(self,attention_size, attention_input_size):
		super(SemanticAttention,self).__init__()
		self.attention_size = attention_size
		
		self.attn1 = nn.Linear(attention_input_size,attention_size)
		self.trainable_weights = nn.Parameter(torch.randn(attention_size))

	def forward(self,key,query):
	
		query = self.attn1(query)
		key = key.permute(1,0,2)
		#print(self.trainable_weights.size())
		#print(key.size())
		#print(query.size())
		return torch.sum(self.trainable_weights.expand_as(key)*torch.tanh(key+query.expand_as(key)),-1).permute(1,0)
		

if __name__ == '__main__':

	postcnn = torch.rand((2,64,14,14))
	attention = torch.rand(128)

	feature_attn = FeatureAttention(64,128)
	spatial_attn = SpatialAttention(64,128)

	feature_attn(postcnn,attention)
	spatial_attn(postcnn,attention)
