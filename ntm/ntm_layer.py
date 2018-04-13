import torch
from torch import nn
from data_gen.plot_data import plot_memory_attention

from ntm.ntm_cell import NTMCell


class NTM(nn.Module):

    def __init__(self, tm_in_dim, tm_output_units, tm_state_units,
                 num_heads, is_cam, num_shift, M):
        """Initialize an NTM Layer.

        :param tm_in_dim: input size.
        :param tm_output_units: output size.
        :param tm_state_units: state size.
        :param num_heads: number of heads.
        :param is_cam: is it content_addressable.
        :param num_shift: number of shifts of heads.
        :param M: Number of slots per address in the memory bank.
        """
        super(NTM, self).__init__()

        # Create the NTM components
        self.NTMCell = NTMCell(tm_in_dim, tm_output_units, tm_state_units,
                               num_heads, is_cam, num_shift, M)

    def forward(self, x, state, wt_dynamic):           # x : batch_size, seq_len, input_size
        output = None
        labelx = ''
        i = 0
        k = 0
        for j in range(x.size()[-2]):
            tm_output, state, wt_dynamic = self.NTMCell(x[..., j, :], state, wt_dynamic)

            # plot attention/memory
            if 0:
                label = 'Writing sequence x' + str(k)
                if x[..., j, 0] == 1 and x[..., j, 1] != 1 :
                    labelx = 'Writing sequence y'+ str(i)
                elif x[..., j, 1] == 1 and x[..., j, 0] != 1:
                    labelx = 'Reading sequence y' + str(i)
                    i = i + 1
                elif x[..., j, 1] == 1 and x[..., j, 0] == 1 and x[..., j+1, 2] != 1:
                    k = k + 1
                    labelx = 'Writing sequence x' + str(k)
                elif x[..., j, 1] == 1 and x[..., j, 0] == 1 and x[..., j+1, 2] == 1:
                    labelx = 'Reading sequences x'

                if labelx:
                    label = labelx

                plot_memory_attention(state[2], state[1], label)

            if tm_output is None:
                continue

            tm_output = tm_output[..., None, :]
            if output is None:
                output = tm_output
                continue

            # concatenate output
            output = torch.cat([output, tm_output], dim=-2)

        return output, state


