"""
Defines attention subunit for mental model
"""

__author__ = "Emre Sevgen"

import torch
import numpy as np
import torch.nn as nn

from miprometheus.models.model import Model

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

	semantic_attn = SemanticAttention(8,128)
	keys = torch.rand((2,4,8))
	query = torch.rand((2,128))
	semantic_attn(keys,query)
	
	
