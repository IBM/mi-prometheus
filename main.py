import torch
from torch import nn
from data_gen.generate_copy import data_generator
from data_gen.init_state import init_state
from ntm.ntm_layer import NTM
import numpy as np
import torch.cuda as cuda

torch.set_num_threads(1)
CUDA = False
# set seed
torch.manual_seed(2)
np.random.seed(0)
if CUDA:
    torch.cuda.manual_seed(2)

# data_gen generator x,y
batch_size = 1
min_len = 1
max_len = 10
bias = 0.5
element_size = 8
num_subseq_max = 4
num_subseq_min = 1

# init state, memory, attention
tm_in_dim = element_size +2
tm_output_units = element_size
tm_state_units = 5
n_heads = 1
M = 10
is_cam = False
num_shift = 3

# To be saved for testing
args_save = {'tm_in_dim': tm_in_dim, 'tm_output_units': tm_output_units, 'tm_state_units': tm_state_units
             , 'n_heads': n_heads, 'is_cam': is_cam, 'num_shift': num_shift, 'M': M, 'element_size': element_size}

# Instantiate
ntm = NTM(tm_in_dim, tm_output_units, tm_state_units, n_heads, is_cam, num_shift, M)
if CUDA:
    ntm.cuda()

# Set loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ntm.parameters(), lr=0.01)

# Start Training
epoch = 0
debug = 10000
valid_steps = 100
active_valid = False
debug_active = 0
# Data generator : input & target
data_gen = data_generator(min_len, max_len, batch_size, bias, element_size, num_subseq_min, num_subseq_max)
for inputs, targets, num_subseq, mask in data_gen:

    # Init state, memory, attention
    N = 60# max(seq_length) + 1
    _, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

    optimizer.zero_grad()

    output, states_test = ntm(inputs, states)

    loss = criterion(output[:, mask, :], targets)

    print(", epoch: %d, loss: %1.5f, N %d " % (epoch + 1, loss, N), "nb_sub_sequences:", num_subseq)

    loss.backward()
    optimizer.step()

    if (num_subseq == 3 and (loss < 1e-5)) or epoch == 8000:
        path = "./Models/"
        # save model parameters
        torch.save(ntm.state_dict(), path+"model_parameters")
        # save initial arguments of ntm
        np.save(path + 'ntm_arguments', args_save)
        break

    if not(epoch % valid_steps) and epoch != 0 and active_valid:
        # test accuracy
        output, states_test = ntm(inputs, states, states[1])
        output = torch.round(output[:, mask, :])
        acc = 1 - torch.abs(output - targets)
        accuracy = acc.mean()
        print("Accuracy: %.6f" % (accuracy * 100) + "%")
        print("nb markers valid", num_subseq)

    epoch += 1

print("Learning finished!")




