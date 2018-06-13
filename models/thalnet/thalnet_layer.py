import torch

from torch import nn
from models.thalnet.thalnet_cell import ThalNetCell
from models.model_base import ModelBase
import logging
import numpy as np
from misc.app_state import AppState


class ThalNet(ModelBase, nn.Module):
    def __init__(self, params):
        """

        :param params:
        """
        self.context_input_size = params['context_input_size']
        self.input_size = params['input_size']
        self.output_size = params['output_size']
        self.center_size = params['num_modules'] * params['center_size_per_module']
        self.center_size_per_module = params['center_size_per_module']
        self.num_modules = params['num_modules']
        self.output_center_size = self.output_size + self.center_size_per_module
        self.app_state = AppState()

        # This is for the time plot
        self.cell_state_history = None

        super(ThalNet, self).__init__()

        # Create the DWM components
        self.ThalnetCell = ThalNetCell(self.input_size, self.output_size, self.context_input_size,
                                       self.center_size_per_module, self.num_modules)

    def forward(self, data_tuple):  # x : batch_size, seq_len, input_size
        """
        Runs the ThalNet cell

        """
        (inputs, _) = data_tuple

        # Set type
        dtype = torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor

        if self.app_state.visualize:
            self.cell_state_history = []

        output = None
        batch_size = inputs.size(0)
        seq_length = inputs.size(-2)

        # init state
        cell_state = self.ThalnetCell.init_state(batch_size, dtype)
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
                self.cell_state_history.append([cell_state[i][0].detach().numpy() for i in range(self.num_modules)] + [cell_state[i][1].hidden_state.detach().numpy() for i in range(self.num_modules)])

        return output

    def generate_figure_layout(self):
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec

        # Change fonts globally - for all figures/subsplots at once.
        rc('font', **{'family': 'Times New Roman'})

        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()

        # Create a specific grid for NTM .
        gs = gridspec.GridSpec(4, 3)

        # modules & centers subplots
        ax_center = [fig.add_subplot(gs[i, 0]) for i in range(self.num_modules)]
        ax_module = [fig.add_subplot(gs[i, 1]) for i in range(self.num_modules)]  #

        # inputs & prediction subplot
        ax_inputs = fig.add_subplot(gs[0, 2])
        ax_pred = fig.add_subplot(gs[2, 2])

        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax_pred.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_pred.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Set labels.
        ax_inputs.set_title('Inputs')
        ax_inputs.set_ylabel('num_row')
        ax_inputs.set_xlabel('num_columns')

        ax_pred.set_title('Prediction')
        ax_pred.set_xlabel('num_classes')

        # centers
        ax_center[0].set_title('center states')
        ax_center[3].set_xlabel('iteration')
        ax_center[0].set_ylabel('center size')
        ax_center[1].set_ylabel('center size')
        ax_center[2].set_ylabel('center size')
        ax_center[3].set_ylabel('center size')

        # modules
        ax_module[0].set_title('module states')
        ax_module[3].set_xlabel('iteration')
        ax_module[0].set_ylabel('module state size')
        ax_module[1].set_ylabel('module state size')
        ax_module[2].set_ylabel('module state size')
        ax_module[3].set_ylabel('module state size')

        # Create buffer and pickle the figure.

        return fig

    def plot_sequence(self, data_tuple, logits):

        num_batch = 0
        (inputs, _) = data_tuple
        inputs = inputs.cpu().detach().numpy()
        predictions_seq = logits.cpu().detach().numpy()

        input_seq = inputs[num_batch, 0] if len(inputs.shape) == 4 else inputs[num_batch]

        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """

        # Create figure template.
        fig = self.generate_figure_layout()
        # Get axes that artists will draw on.

        # Set intial values of displayed  inputs, targets and predictions - simply zeros.
        inputs_displayed = np.zeros(input_seq.shape)

        # Define Modules
        module_state_displayed_1 = np.zeros((self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_2 = np.zeros((self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_3 = np.zeros((self.cell_state_history[0][4].shape[-1], input_seq.shape[-2]))
        module_state_displayed_4 = np.zeros((self.cell_state_history[0][-1].shape[-1], input_seq.shape[-2]))

        # Define centers
        center_state_displayed_1 = np.zeros((self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_2 = np.zeros((self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_3 = np.zeros((self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))
        center_state_displayed_4 = np.zeros((self.cell_state_history[0][0].shape[-1], input_seq.shape[-2]))

        #modules_plot = [module_state_displayed for _ in range(self.num_modules)]
        #center_plot = [center_state_displayed for _ in range(self.num_modules)]

        # Set initial values of memory and attentions.
        # Unpack initial state.

        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))

        # Create frames - a list of lists, where each row is a list of artists used to draw a given frame.
        frames = []

        for i, (input_element, state_tuple) in enumerate(
                zip(input_seq, self.cell_state_history)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0] // 10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))

            # Update displayed values on adequate positions.
            inputs_displayed[i, :] = input_element

            # Create "Artists" drawing data on "ImageAxes".
            artists = [None] * len(fig.axes)

            # centers state
            center_state_displayed_1[:, i] = state_tuple[0][num_batch, :]
            entity = fig.axes[0]
            artists[0] = entity.imshow(center_state_displayed_1, interpolation='nearest', aspect='auto')

            center_state_displayed_2[:, i] = state_tuple[1][num_batch, :]
            entity = fig.axes[1]
            artists[1] = entity.imshow(center_state_displayed_2, interpolation='nearest', aspect='auto')

            center_state_displayed_3[:, i] = state_tuple[2][num_batch, :]
            entity = fig.axes[2]
            artists[2] = entity.imshow(center_state_displayed_3, interpolation='nearest', aspect='auto')

            center_state_displayed_4[:, i] = state_tuple[3][num_batch, :]
            entity = fig.axes[3]
            artists[3] = entity.imshow(center_state_displayed_4, interpolation='nearest', aspect='auto')

            # module state
            module_state_displayed_1[:, i] = state_tuple[4][num_batch, :]
            entity = fig.axes[4]
            artists[4] = entity.imshow(module_state_displayed_1, interpolation='nearest', aspect='auto')

            module_state_displayed_2[:, i] = state_tuple[5][num_batch, :]
            entity = fig.axes[5]
            artists[5] = entity.imshow(module_state_displayed_2, interpolation='nearest', aspect='auto')

            module_state_displayed_3[:, i] = state_tuple[6][num_batch, :]
            entity = fig.axes[6]
            artists[6] = entity.imshow(module_state_displayed_3, interpolation='nearest', aspect='auto')

            module_state_displayed_4[:, i] = state_tuple[7][num_batch, :]
            entity = fig.axes[7]
            artists[7] = entity.imshow(module_state_displayed_4, interpolation='nearest', aspect='auto')

            # h = 0
            # for j, state in enumerate(state_tuple):
            #     # Get attention of head 0.
            #
            #     # "Show" data on "axes".
            #     entity = fig.axes[j]
            #     if self.num_modules <= h < 2 * self.num_modules :
            #         modules_plot[j - self.num_modules][:, i] = state[num_batch, :]
            #         artists[j] = entity.imshow(modules_plot[j - self.num_modules], interpolation='nearest', aspect='auto')
            #
            #     else:
            #         center_plot[j][:, i] = state[num_batch, :]
            #         artists[j] = entity.imshow(center_plot[j], interpolation='nearest', aspect='auto')
            #
            #     h += 1

            entity = fig.axes[2 * self.num_modules]
            artists[2 * self.num_modules] = entity.imshow(inputs_displayed, interpolation='nearest', aspect='auto')

            entity = fig.axes[2 * self.num_modules + 1]
            artists[2 * self.num_modules + 1] = entity.imshow(predictions_seq[0, -1, None], interpolation='nearest', aspect='auto')

            # Add "frame".
            frames.append(artists)

        # print("--- %s seconds ---" % (time.time() - start_time))
        # Update time plot fir generated list of figures.
        self.plot.update(fig,  frames)
        return self.plot.is_closed


if __name__ == "__main__":
    input_size = 28
    params = {'context_input_size': 32, 'input_size' : input_size, 'output_size': 10,
              'center_size': 1, 'center_size_per_module':32 , 'num_modules':4}

    # Initialize the application state singleton.
    app_state = AppState()
    app_state.visualize = True

    model = ThalNet(params)

    seq_length = 10
    batch_size = 2

    # Check for different seq_lengts and batch_sizes.
    for i in range(1):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, 1, input_size, input_size)
        logits = torch.randn(batch_size, 1, params['output_size'])
        y = x
        data_tuple = (x, y)

        # Test forward pass.
        y_pred = model(data_tuple)

        app_state.visualize = True
        if app_state.visualize:
            model.plot_sequence(data_tuple, logits)

        # Change batch size and seq_length.
        seq_length = seq_length + 1
        batch_size = batch_size + 1



