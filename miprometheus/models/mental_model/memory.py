import torch


class Memory(object):
	def __init__(self, batch_size,mem_slots,object_size,controller_out_size):
		# memory is [batch size x memory slots x object size]
		self.memory = torch.zeros((batch_size,mem_slots,object_size))
		
		# generate key, subset, location address and address mixing for read 
		self.read_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.read_subset_gen= torch.nn.Linear(controller_out_size, object_size)
		self.read_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.read_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.read_gate = torch.nn.Linear(controller_out_size,object_size)

		# generate key, subset, location address and address mixing for erase
		self.erase_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.erase_subset_gen = torch.nn.Linear(controller_out_size, object_size)
		self.erase_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.erase_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.erase_gate = torch.nn.Linear(controller_out_size,object_size)

		# generate key, subset, location address and address mixing for write
		self.write_keygen = torch.nn.Linear(controller_out_size, object_size)
		self.write_subset_gen = torch.nn.Linear(controller_out_size, object_size)
		self.write_mix_gen= torch.nn.Linear(controller_out_size, 2)
		self.write_location = torch.nn.Linear(controller_out_size, mem_slots)
		self.write_gate = torch.nn.Linear(controller_out_size,object_size)

	def subset_similarity(self, key, subset_attention):
		# Loop over objects in memory
		# Determine similarity between key and object, but weighed
		# by the subset attention. Essentially only compare similarity
		# to attended dimensions
		
		# key is [batch size x object size]
		# subset attention is [batch size x object size]
		# output is [batch size x memory slots]

		key = torch.nn.functional.normalize(key,dim=-1)
		norm_mem = torch.nn.functional.normalize(self.memory,dim=-1)
		content_address = torch.sum(key.unsqueeze(1) * norm_mem * subset_attention.unsqueeze(1),-1)

		return content_address

	def address_mix(self,content_address,location_address,mix):
		return content_address*mix[:,0].unsqueeze(-1) + location_address*mix[:,1].unsqueeze(-1)

	def read(self,location, gate):
		# Read object given location

		# location is [batch size x memory slots]

		return torch.sum(self.memory * location.unsqueeze(2) * gate.unsqueeze(1),1)

	def erase(self, location, gate):
		# Erase from location by multiplying by erase vector
		# Erase vector has values ranging from 0 to 1 for each entry
		# Location vector refers to memory slot

		# location is [batch size x memory slots]
		# erase is [batch size x object size]

		self.memory = self.memory - self.memory * (location.unsqueeze(2) * gate.unsqueeze(1))

	def write(self, location, gate, obj):
		# Add to location the obj vector
		
		# location is [batch size x memory slots]
		# obj is [batch size x object size]

		self.memory = self.memory + (location.unsqueeze(2) * obj.unsqueeze(1) *gate.unsqueeze(1))
		

if __name__ == '__main__':
	mem = Memory(2,2,5)
	location = torch.Tensor([[1,0],[0,1]])
	obj = torch.Tensor([[1,0,0,0,0],[0,1,0,0,0]])
	mem.write(location,obj)
	print(mem.memory)

	erase = torch.Tensor([[1,1,1,1,1],[1,1,1,1,1]])
	mem.erase(location,erase)
	print(mem.memory)
	
