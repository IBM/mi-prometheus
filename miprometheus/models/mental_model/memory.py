import torch
import torch.nn as nn


class Memory(nn.Module):
	"""
	``Memory`` defines the memory subunit for mental model prototype. Memory has functions to store
	objects, compare similarity to keys subject to gating, read, erase and write functions with gates.

	"""
	def __init__(self,mem_slots,object_size,controller_out_size,app_state):
		"""
		Constructor of the ``Memory`` subunit

		:param mem_slots: Number of slots in memory for objects.
		:type mem_slots: Int

		:param object_size: Size of objects to be stored.
		:type object_size: Int

		:param controller_out_size: Size of objects to be stored.
		:type controller_out_size: Int

		:param app_state: For enabling CUDA types when running on GPU.
		:type app_state: AppState

		"""
		super(Memory,self).__init__()

		# Set constants for memory reset
		self.mem_slots = mem_slots
		self.object_size = object_size

		# Get app state gpu/cpu
		self.dtype = app_state.dtype
		
		# generate key, subset, location address, address mixing and sharpening for read 
		self.read_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.read_subset_gen= torch.nn.Linear(controller_out_size, object_size)
		self.read_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.read_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.read_gate = torch.nn.Linear(controller_out_size,object_size)
		self.read_sharpen = torch.nn.Linear(controller_out_size,1)

		# generate key, subset, location address and address mixing for erase
		self.erase_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.erase_subset_gen = torch.nn.Linear(controller_out_size, object_size)
		self.erase_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.erase_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.erase_gate = torch.nn.Linear(controller_out_size,object_size)
		self.erase_sharpen = torch.nn.Linear(controller_out_size,1)

		# generate key, subset, location address and address mixing for write
		self.write_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.write_subset_gen = torch.nn.Linear(controller_out_size, object_size)
		self.write_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.write_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.write_gate = torch.nn.Linear(controller_out_size,object_size)
		self.write_sharpen = torch.nn.Linear(controller_out_size,1)


		# Initialize all layers
		# reads
		nn.init.xavier_uniform_(self.read_keygen.weight)
		nn.init.xavier_uniform_(self.read_subset_gen.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.read_mix_gen.weight)
		nn.init.xavier_uniform_(self.read_location.weight)
		nn.init.xavier_uniform_(self.read_gate.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.read_sharpen.weight)

		self.read_keygen.bias.data.fill_(0.01)
		self.read_subset_gen.bias.data.fill_(0.01)
		self.read_mix_gen.bias.data.fill_(0.01)
		self.read_location.bias.data.fill_(0.01)
		self.read_gate.bias.data.fill_(0.01)
		self.read_sharpen.bias.data.fill_(1.00)

		# erases
		nn.init.xavier_uniform_(self.erase_keygen.weight)
		nn.init.xavier_uniform_(self.erase_subset_gen.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.erase_mix_gen.weight)
		nn.init.xavier_uniform_(self.erase_location.weight)
		nn.init.xavier_uniform_(self.erase_gate.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.erase_sharpen.weight)

		self.erase_keygen.bias.data.fill_(0.01)
		self.erase_subset_gen.bias.data.fill_(0.01)
		self.erase_mix_gen.bias.data.fill_(0.01)
		self.erase_location.bias.data.fill_(0.01)
		self.erase_gate.bias.data.fill_(0.01)
		self.erase_sharpen.bias.data.fill_(1.00)

		# writes
		nn.init.xavier_uniform_(self.write_keygen.weight)
		nn.init.xavier_uniform_(self.write_subset_gen.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.write_mix_gen.weight)
		nn.init.xavier_uniform_(self.write_location.weight)
		nn.init.xavier_uniform_(self.write_gate.weight, gain=nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_uniform_(self.write_sharpen.weight)

		self.write_keygen.bias.data.fill_(0.01)
		self.write_subset_gen.bias.data.fill_(0.01)
		self.write_mix_gen.bias.data.fill_(0.01)
		self.write_location.bias.data.fill_(0.01)
		self.write_gate.bias.data.fill_(0.01)
		self.write_sharpen.bias.data.fill_(1.00)

	def subset_similarity(self, key, subset_attention):
		"""
		Returns the similarity of a key to objects in memory.
		Only compares a subset of the full object vector.

		:param key: Key to compare similarity to.
		:type key: torch.Tensor

		:param subset_attention: Subset of the object and key vectors to compare.
		:type subset_attention: torch.Tensor

		"""
		# Loop over objects in memory
		# Determine similarity between key and object, but weighed
		# by the subset attention. Essentially only compare similarity
		# to attended dimensions
		
		# key is [batch size x object size]
		# subset attention is [batch size x object size]
		# output is [batch size x memory slots]

		key = torch.nn.functional.normalize(key,dim=-1)
		norm_mem = torch.nn.functional.normalize(self.memory + 1e-12,dim=-1)
		content_address = torch.sum(key.unsqueeze(1) * norm_mem * subset_attention.unsqueeze(1),-1)

		return content_address

	def address_mix(self,content_address,location_address,mix):
		"""
		Takes address based on content and location, and returns one combined attention over memory.

		:param content_address: Content-based attention tensor.
		:type content_address: torch.Tensor

		:param location_address: Location-based attention tensor.
		:type location_address: torch.Tensor

		:param mix: Mixing ratio.
		:type mix: torch.Tensor

		"""
		return content_address*mix[:,0].unsqueeze(-1) + location_address*mix[:,1].unsqueeze(-1)

	def read(self,location, gate):
		"""
		Read object given attention, subject to gate.

		:param location: Attention over memory.
		:type location: torch.Tensor

		:param gate: Gating of the read. Allows passing of only a subset of the object.
		:type gate: torch.Tensor

		"""
		# Read object given location

		# location is [batch size x memory slots]

		return torch.sum(self.memory * location.unsqueeze(2) * gate.unsqueeze(1),1)

	def erase(self, location, gate):
		"""
		Erase from memory given attention, subject to gate.

		:param location: Attention over memory.
		:type location: torch.Tensor

		:param gate: Gating of the erase. Allows erasing of only a subset of the object.
		:type gate: torch.Tensor

		"""
		# Erase from location by multiplying by erase vector
		# Erase vector has values ranging from 0 to 1 for each entry
		# Location vector refers to memory slot

		# location is [batch size x memory slots]
		# erase is [batch size x object size]

		#print(self.memory,'pre erase')
		#print('loc',location)
		#print('gate',gate)
		self.memory = self.memory - self.memory * (location.unsqueeze(2) * gate.unsqueeze(1))
		#print(self.memory,'post erase')

	def write(self, location, gate, obj):
		"""
		Write memory given attention and object, subject to gate.

		:param location: Attention over memory.
		:type location: torch.Tensor

		:param gate: Gating of the write. Allows writing of only a subset of the object.
		:type gate: torch.Tensor

		:param obj: Object to write to memory.
		:type obj: torch.Tensor

		"""
		# Add to location the obj vector
		
		# location is [batch size x memory slots]
		# obj is [batch size x object size]

		self.memory = self.memory + (location.unsqueeze(2) * obj.unsqueeze(1) *gate.unsqueeze(1))

	def reset(self,batch_size):
		"""
		Reinitialized memory to all zeros given batch size.

		:param batch_size: Size of current batch.
		:type batch_size: Int

		"""
		# Reset memory to all zeros.

		# memory is [batch size x memory slots x object size]
		self.memory = torch.zeros((batch_size,self.mem_slots,self.object_size)).type(self.dtype)

	def sharpen(self,weights,sharpening):
		"""
		Sharpens attention (weights) given sharpening factor.

		:param weights: Attention tensor.
		:type weights: torch.Tensor

		:param sharpening: Amount of sharpening.
		:type sharpening: float

		"""
		# Sharpen the weight with sharpening factor given by the head

		# Weights is [batch size x memory slots]
		# sharpening is [batch size x 1]
		sharpening, _ = torch.max(sharpening,1)
		weights = torch.pow(weights + 1e-12, sharpening.unsqueeze(1))
		weights = torch.nn.functional.normalize(weights, p=1, dim=1)
		return weights
		

if __name__ == '__main__':
	from miprometheus.utils.app_state import AppState
	app_state = AppState()
	mem = Memory(2,5,5,app_state)
	location = torch.Tensor([[1,0],[0,1]])
	obj = torch.Tensor([[1,0,0,0,0],[0,1,0,1,0]])
	gate = torch.Tensor([[1,1,0,0,0],[1,0,0,1,0]])
	mem.reset(1)
	mem.write(location,gate,obj)
	print(mem.memory)

	erase = torch.Tensor([[1,1,1,1,1],[1,1,1,1,1]])
	mem.erase(location,erase)
	print(mem.memory)
	
