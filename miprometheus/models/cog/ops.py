"""
Defines attention subunits of the COG model
"""

__author__ = "Emre Sevgen"

import torch
import torch.nn as nn

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
		
		# Define the scaling and shifting network, initialize to 1.
		self.attn1 = nn.Linear(attention_input_size,attention_size*2)
		
		# Initialize network
		nn.init.xavier_uniform_(self.attn1.weight, gain=nn.init.calculate_gain('relu'))
		self.attn1.bias.data.fill_(0.0)

	def forward(self,inputs_to_attend,attn_gen):
	
		shift_and_scale = self.attn1(attn_gen)
		shift, scale = torch.chunk(shift_and_scale,2,-1)
		
		scale = nn.functional.relu(scale + 1.0)

		inputs_to_attend = inputs_to_attend + shift.view(-1,self.attention_size,1,1)
		inputs_to_attend = inputs_to_attend * scale.view(-1,self.attention_size,1,1)
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
		self.attn2 = nn.Linear(10,attention_size[0]*attention_size[1])

		# Initialize network
		nn.init.xavier_uniform_(self.attn1.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.attn2.weight)
		self.attn1.bias.data.fill_(0.0)
		self.attn2.bias.data.fill_(0.0)
	
	def forward(self,inputs_to_attend,attn_gen):
		attn_gen = nn.functional.relu(self.attn1(attn_gen))
		attn_gen = nn.functional.softmax(self.attn2(attn_gen),dim=-1)

		inputs_to_attend *= attn_gen.view(-1,1,self.attention_size[0],self.attention_size[1])

		return inputs_to_attend, attn_gen


class SemanticAttention(nn.Module):
	def __init__(self,attention_size, attention_input_size):
		super(SemanticAttention,self).__init__()
		self.attention_size = attention_size
		
		self.attn1 = nn.Linear(attention_input_size,attention_size)
		self.trainable_weights = nn.Parameter(torch.randn(attention_size)*0.1)

		# Initialize network
		nn.init.xavier_uniform_(self.attn1.weight)
		self.attn1.bias.data.fill_(0.0)

	def forward(self,keys,query):
	
		query = self.attn1(query)
		#print(keys.size())
		#print(self.trainable_weights.size())
		#print(query.size())
		# keys is batch x sequence x embedding length
		# trainable weights is embedding length
		# query is batch x embedding length
		# similarity is batch x sequence

		similarity = torch.sum(
		self.trainable_weights.unsqueeze(0).unsqueeze(0) * 
		torch.tanh(query.unsqueeze(1) + keys), -1 )

		post_attention = torch.sum(
		keys * similarity.unsqueeze(-1), 1)

		return post_attention
		

if __name__ == '__main__':

	postcnn = torch.rand((1,4,2,2))
	attention = torch.rand(128)

	feature_attn = FeatureAttention(4,128)
	spatial_attn = SpatialAttention([2,2],128)

	feature_attn(postcnn,attention)
	spatial_attn(postcnn,attention)

	semantic_attn = SemanticAttention(8,128)
	keys = torch.rand((2,4,8))
	query = torch.rand((2,128))
	semantic_attn(keys,query)
	
	
