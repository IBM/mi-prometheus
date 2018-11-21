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

	def forward(self,inputs,state=None,controls=None):

		if controls is None:
			controls = torch.randn(self.control_input_size)

		if state is None:
			state = torch.zeros(self.n_maps,self.shape[0],self.shape[1])

		# Generate gate parameters from control input via linear layer			
		inputs_control = self.control1(controls)

		# Split gate inputs into respective gates
		f, i, o = torch.split(inputs_control,(self.num_units_forget_gate, 
																					self.num_units_input_gate,
																					self.num_units_output_gate))

		in_gates = i.view(self.n_maps,self.in_channels,1,1)
		# I think TF has separable convolution
		gated_inputs = nn.functional.conv2d(inputs,in_gates)

		# TensorFlow implementation has bias.
		forget_gates = f.view(self.n_maps,1,1)
		gated_states = state * forget_gates
		new_state = gated_inputs + gated_states

		output_gates = o.view(self.out_channels,self.n_maps,1,1)
		# Again, I think TF uses separable convolution
		output = torch.tanh(new_state)
		output = nn.functional.conv2d(output,output_gates)

		return output, new_state

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
	
