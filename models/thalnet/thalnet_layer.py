import torch

from torch import nn
from models.thalnet.thalnet_cell import ThalNetCell
from models.model_base import ModelBase
import pickle
import io
import logging
import numpy as np
from misc.app_state import AppState

class THALNET(ModelBase, nn.Module):

    def __init__(self, params):
        """Initialize an THALNET Layer. """
        self.context_input_size = params['context_input_size']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.center_size = params['num_modules'] * params['center_size_per_module']
        self.center_size_per_module = params['center_size_per_module']
        self.num_modules = params['num_modules']
        self.output_center_size =  self.output_size + self.center_size_per_module
        self.app_state = AppState()

        # This is for the time plot
        self.cell_state_history = None

        super(THALNET, self).__init__()

        # Create the DWM components
        self.ThalnetCell = ThalNetCell(self.input_size, self.output_size, self.context_input_size,
                                       self.center_size_per_module, self.num_modules)

    def forward(self, data_tuple):  # x : batch_size, seq_len, input_size
        """
        Runs the DWM cell and plots if necessary

        :param x: input sequence  [BATCH_SIZE x seq_len x input_size ]
        :param state: Input hidden state  [BATCH_SIZE x state_size]
        :return: Tuple [output, hidden_state]
        """
        (inputs, _) = data_tuple

        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(-2)

        # init state
        cell_state = self.init_state(batch_size)
        for j in range(seq_length):
            output_cell, cell_state = self.ThalnetCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell

            # concatenate output
            else:
                output = torch.cat([output, output_cell], dim=-2)

            # This is for the time plot
            if self.app_state.visualize:
                self.cell_state_history.append((cell_state[0].detach().numpy(),
                                                cell_state[1].detach().numpy(),
                                                cell_state[2].detach().numpy(),
                                                cell_state[3].detach().numpy(),
                                                cell_state[4].detach().numpy(),
                                                cell_state[5].detach().numpy(),
                                                cell_state[6].detach().numpy(),
                                                cell_state[7].detach().numpy()
                                                ))

        return output

    def init_state(self, batch_size):

        # center state initialisation
        center_state_per_module = [torch.randn((batch_size, self.center_size_per_module))
                                        for _ in range(self.num_modules)]

        # module state initialisation
        module_states = [torch.randn((batch_size, self.center_size_per_module if i != self.num_modules - 1 else self.output_center_size))
                         for i in range(self.num_modules)]

        states = center_state_per_module + module_states

        return states

    def pickle_figure_template(self):
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec

        # Change fonts globally - for all figures/subsplots at once.
        rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()
        # axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'width_ratios': [input_seq.shape[0]]})

        # Create a specific grid for NTM .
        gs = gridspec.GridSpec(4, 3)

        # module 1
        ax_center = [fig.add_subplot(gs[i, 0]) for i in range(self.num_modules)]
        ax_module = [fig.add_subplot(gs[i, 1]) for i in range(self.num_modules)]  #

        ax_inputs = fig.add_subplot(gs[0, 2])

        ax_pred = fig.add_subplot(gs[1, 2])


        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # module 1
        # ax_center_1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_center_1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #
        # # module 2
        # ax_center_2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_center_2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #
        # # module 3
        # ax_center_3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_center_3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #
        # # module 4
        # ax_center_4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_center_4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        # ax_module_4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Set labels.
        ax_inputs.set_title('Inputs')
        ax_inputs.set_ylabel('num_row')

        # module 1
        # ax_center_1.set_title('center state module 1')
        # ax_center_1.set_ylabel('center size')
        # ax_center_1.set_xlabel('iteration')
        # ax_module_1.set_title('state module 1')
        # ax_module_1.set_xlabel('Iteration')
        #
        # # module 2
        # ax_center_2.set_title('center state module 1')
        # ax_center_2.set_ylabel('center size')
        # ax_center_2.set_xlabel('iteration')
        # ax_module_2.set_title('state module 1')
        # ax_module_2.set_xlabel('Iteration')
        #
        # # module 3
        # ax_center_3.set_title('center state module 1')
        # ax_center_3.set_ylabel('center size')
        # ax_center_3.set_xlabel('iteration')
        # ax_module_3.set_title('state module 1')
        # ax_module_3.set_xlabel('Iteration')
        #
        # # module 4
        # ax_center_4.set_title('center state module 1')
        # ax_center_4.set_ylabel('center size')
        # ax_center_4.set_xlabel('iteration')
        # ax_module_4.set_title('state module 1')
        # ax_module_4.set_xlabel('Iteration')

        # Create buffer and pickle the figure.
        buf = io.BytesIO()
        pickle.dump(fig, buf)

        return buf

    def plot_sequence(self, logits, data_tuple):

        num_batch = 0
        (inputs, _) = data_tuple
        input_seq = inputs[num_batch, 0] if len(inputs.shape) == 4 else inputs[num_batch]

        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        # import time
        # start_time = time.time()
        # Create figure template.
        buf = self.pickle_figure_template()

        # Set intial values of displayed  inputs, targets and predictions - simply zeros.
        inputs_displayed = np.transpose(np.zeros(input_seq.shape))

        module_state_displayed = np.zeros((self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed = np.zeros((self.cell_state_history[0][1].shape[-1], input_seq.shape[-2]))

        module_state_displayed_t = np.zeros((self.cell_state_history[0][7].shape[-1], input_seq.shape[-2]))

        # Set initial values of memory and attentions.
        # Unpack initial state.

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))
        # List of figures.
        figs = []
        for i, (input_element, state_tuple) in enumerate(
                zip(input_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0] // 10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))

            # Create figure object on the basis of template.
            buf.seek(0)
            fig = pickle.load(buf)

            # Update displayed values on adequate positions.
            inputs_displayed[:, i] = input_element

            h = 0
            for j, state in enumerate(state_tuple):
                # Get attention of head 0.

                # "Show" data on "axes".
                entity = fig.axes[j]
                if self.num_modules <= h < 2 * self.num_modules :
                    if h != 2 * self.num_modules - 1:
                        module_state_displayed[:, i] = state[num_batch, :]
                        entity.imshow(module_state_displayed, interpolation='nearest', aspect='auto')
                    else:
                        module_state_displayed_t[:, i] = state[num_batch, :]
                        entity.imshow(module_state_displayed_t, interpolation='nearest', aspect='auto')

                else:
                    center_state_displayed[:, i] = state[num_batch, :]
                    entity.imshow(center_state_displayed, interpolation='nearest', aspect='auto')

                h += 1

            entity = fig.axes[2 * self.num_modules]
            entity.imshow(inputs_displayed, interpolation='nearest', aspect='auto')

            entity = fig.axes[2 * self.num_modules + 1]
            entity.imshow(logits[0, -1, None], interpolation='nearest', aspect='auto')

            # Append figure to a list.
            fig.set_tight_layout(True)
            figs.append(fig)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Update time plot fir generated list of figures.
        self.plot.update(figs)
        return self.plot.is_closed


if __name__ == "__main__":
    input_size = 28
    params = {'context_input_size': 32, 'input_size' : input_size, 'output_size': 10,
              'center_size': 1, 'center_size_per_module':32 , 'num_modules':4}

    # Initialize the application state singleton.
    app_state = AppState()
    app_state.visualize = True

    model = THALNET(params)

    seq_length = 10
    batch_size = 2

    # Check for different seq_lengts and batch_sizes.
    for i in range(2):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, 1, input_size, input_size)
        logits = torch.randn(batch_size, 1, params['output_size'])
        y = x
        data_tuple = (x, y)


        # Test forward pass.
        y_pred = model(data_tuple)

        app_state.visualize = True
        if app_state.visualize:
            model.plot_sequence(logits, data_tuple)

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1



