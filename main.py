import torch
from torch import nn
from data_gen.build_data_distraction import init_state, build_data_gen
from ntm.ntm_layer import NTM
from data_gen.plot_data import plot_memory_attention
import numpy as np
from torch.autograd import Variable
import torch.cuda as cuda

CUDA = False
# set seed
torch.manual_seed(2)
np.random.seed(0)
if CUDA:
    torch.cuda.manual_seed(2)

# data_gen generator x,y
batch_size = 1
min_len = 5
max_len = 15
bias = 0.5
element_size = 8
nb_markers_max = 4

# init state, memory, attention
tm_in_dim = element_size + 3
tm_output_units = element_size
tm_state_units = 2
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
#optimizer = torch.optim.RMSprop(ntm.parameters(), lr=0.001, momentum=0.9, alpha=0.95)

# Start Training
epoch = 0
debug = 10000
valid = 100
debug_active = 0
# Data generator : input & target
data_gen = build_data_gen(min_len, max_len, batch_size, bias, element_size, nb_markers_max)
for inputs, targets, seq_length in data_gen:

    # Init state, memory, attention
    N = 80# max(seq_length) + 1
    _, states = init_state(batch_size, tm_output_units, tm_state_units, n_heads, N, M)

    optimizer.zero_grad()

    output, states_test = ntm(inputs, states)
    loss = criterion(output[:, -(sum(seq_length)):, :], targets)

    print(", epoch: %d, loss: %1.5f, N %d " % (epoch + 1, loss, N), "seq_lengths:", seq_length)

    loss.backward()
    optimizer.step()

    if not(epoch % debug) and epoch != 0 and debug_active:
        plot_memory_attention(states_test[2], states_test[1])
        print(states_test[1])
        debug = 50

    if (epoch==40000) or (loss < 1e-5 and len(seq_length)>=2):
        print("convergence of sequence type:", len(seq_length)-1)
        path = "./Models/"
        # save model parameters
        torch.save(ntm.state_dict(), path+"model_parameters")
        # save initial arguments of ntm
        np.save(path + 'ntm_arguments', args_save)
        break

    if not(epoch % valid) and epoch != 0:
        # test accuracy
        output = torch.round(output[:, -(sum(seq_length)):, :])
        acc = 1 - torch.abs(output - targets)
        accuracy = acc.mean()
        print("Accuracy: %.6f" % (accuracy * 100) + "%")

    epoch += 1

print("Learning finished!")


#ok I see...thank u once again for your effort


