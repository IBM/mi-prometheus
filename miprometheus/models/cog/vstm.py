import torch
import torch.nn as nn

# Was initially inheriting from nn.LSTM, but for now use Module



# Overall, accepts post-attention processed image, for example: 
# (32, 64, 14, 14) [batch,channels,kernel,kernel]

# Spits out a tensor that's (I think) the same size.

class VSTM(nn.Module):
	def __init__(	self, 
								shape,
								in_channels,
								out_channels,
								n_maps,
								control_input_size								
								):
		super(VSTM, self).__init__()

		# Shape of each visual memory map
		self.shape = shape
		# Number of input maps or channels
		self.in_channels = in_channels
		# Number of output maps to project to
		self.out_channels = out_channels
		# Number of maps in the visual memory cell
		self.n_maps = n_maps
		# Lenght of the instruction to be received
		self.control_input_size = control_input_size


		# Calculate number of units for each gate
		self.num_units_forget_gate = n_maps
		self.num_units_input_gate = in_channels * n_maps
		self.num_units_output_gate = out_channels * n_maps

		self.state_split = (self.num_units_forget_gate,
		                     self.num_units_input_gate,
		                     self.num_units_output_gate)

		self.control_output_size = int(self.num_units_forget_gate+self.num_units_input_gate+self.num_units_output_gate)

		# Initialize linear layer that generates controls
		# Run an input control instruction from a source through a linear layer
		# to produce a correctly sized control output. This goes into the
		# Forget, Input and Output gates.
		# Should it have bias?
		self.control1 = nn.Linear(self.control_input_size, self.control_output_size,bias=False)

	def forward(self,inputs,state,controls,dtype):


		#print('VSTM Controls size: {}'.format(controls.size()))
		#print('VSTM State size: {}'.format(state.size()))

		# Generate gate parameters from control input via linear layer			
		inputs_control = self.control1(controls)

		# Split gate inputs into respective gates
		f, i, o = torch.split(inputs_control,(self.num_units_forget_gate, 
																					self.num_units_input_gate,
																					self.num_units_output_gate),-1)


		in_gates = i.view(-1,self.n_maps,self.in_channels,1,1)
		forget_gates = f.view(-1,self.n_maps,1,1)
		output_gates = o.view(-1,self.out_channels,self.n_maps,1,1)
		

		# Probably inefficient, but that's ok for now
		gated_inputs = torch.zeros(inputs.size()[0],self.n_maps,self.shape[0],self.shape[1]).type(dtype)		
		for i in range(inputs.size(0)):
			gated_inputs[i:i+1] = nn.functional.conv2d(inputs[i:i+1],in_gates[i])

		# TensorFlow implementation has bias.
		gated_states = state * forget_gates
		new_state = gated_inputs + gated_states

		# Probably inefficient, but that's ok for now
		output = torch.tanh(new_state)
		outputs = torch.zeros(inputs.size(0),self.out_channels,self.shape[0],self.shape[1]).type(dtype)
		for i in range(inputs.size(0)):
			# Is this correct?
			outputs[i:i+1] = nn.functional.conv2d(output[i:i+1],output_gates[i])

		return outputs, new_state

if __name__ == '__main__':
	# Test with:
		# batch = 2
		# shape = (14,14)
		# in_channels = 64
		# out_channels = 3
		# n_maps = 4
		# control_input_size = 128
	postcnn = torch.rand((2,64,14,14))
	control = torch.rand(128)
	state = torch.zeros((4,14,14))
	vstm = VSTM((14,14),64,3,4,128)
	
	output, new_state = vstm(postcnn)
	output2, new_state = vstm(postcnn,new_state,control)
	print(output2.size())
	print(new_state.size())
	
