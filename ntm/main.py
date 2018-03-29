import torch
from torch import nn
from ntm.build_data_copy import init_state, build_data_gen
import pdb
from ntm.ntm_layer import NTM

# data generator x,y
batch_size = 1
min_len = 10
max_len = 20
bias = 0.5
element_size = 8

# init state, memory, attention
tm_in_dim = element_size + 1
tm_output_units = element_size
tm_state_units = 4
n_heads = 2
N = 30
M = 10
is_cam = False
num_shift = 3

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units,tm_state_units, n_heads, is_cam, num_shift, M)

# Init state, memory, attention
tm_output, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

# Set loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ntm.parameters(), lr=0.01)

# Training
epoch = 0
Train = True
while Train:
    # Data generator : input & target
    data_gen = build_data_gen(min_len, max_len, batch_size, bias, element_size)

    for inputs, targets, seq_length in data_gen:
        optimizer.zero_grad()

        output, _ = ntm(inputs, states)
        loss = criterion(output[:, -seq_length:, :], targets)

        print(", epoch: %d, loss: %1.3f, seq_length %d" % (epoch + 1, loss, seq_length))

        loss.backward()
        optimizer.step()

        #if loss < 1e-5 and inputs.size()[1] == 2*seq_length:
        #    print("Task 1 converged")

        if loss < 1e-5: #and inputs.size()[1] > 2*seq_length:
            Train = False
            Path = "/Users/younesbouhadajr/Documents/Neural_Network/working_memory/Models/model"
            torch.save(ntm.state_dict(), Path)
        epoch += 1

print("Learning finished!")

# Initial Test
print("Testing")
print("input")

min_len = 80
max_len = 80
N = 100
# New sequence
data_gen = build_data_gen(min_len, max_len, batch_size, bias, element_size)
_, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

for inputs, targets, seq_length in data_gen:
    output, states = ntm(inputs, states)
    print("output", output[:, -seq_length:, :])
    print("targets", targets)
    print("memory", states[2])

    # test accuracy
    output = torch.round(output[:, -seq_length:, :])
    acc = 1 - torch.abs(output-targets)
    accuracy = acc.mean()
    print("Accuracy: %.6f" % (accuracy * 100) + "%")




