#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_module.py: pytorch module implementing Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import logging
import numpy as np

# Fix so we can call 
import os,  sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..')) # add path to main project directory.
from misc.app_state import AppState
from models.model_base import ModelBase
from models.ntm.ntm_cell import NTMCell

class NTM(ModelBase, torch.nn.Module):
    '''  Class representing the Neural Turing Machine module. '''
    def __init__(self, params):
        '''
        Constructor. Initializes parameters on the basis of dictionary of parameters passed as argument.
        
        :param params: Dictionary of parameters.
        '''
        # Call constructor of base class.
        super(NTM, self).__init__() 

        # Parse parameters.
        # It is stored here, but will we used ONLY ONCE - for initialization of memory called from the forward() function.
        self.num_memory_addresses = params['memory']['num_addresses']
        
        # Initialize recurrent NTM cell.
        self.ntm_cell = NTMCell(params)
        
    def forward(self, inputs_BxSxI):
        """
        Forward function accepts a Tensor of input data of size [BATCH_SIZE x LENGTH_SIZE x INPUT_SIZE] and 
        outputs a Tensor of size  [BATCH_SIZE x LENGTH_SIZE x OUTPUT_SIZE] . 
        """
        
        # "Data-driven memory size" - temporal solution.
        # Check memory size.
        num_memory_addresses = self.num_memory_addresses
        if num_memory_addresses == -1:
            # Set equal to input sequence length.
            num_memory_addresses = inputs_BxSxI.size(1)
        
        # Initialize 'zero' state.
        cell_state = self.ntm_cell.init_state(inputs_BxSxI.size(0),  num_memory_addresses)

        # List of output logits [BATCH_SIZE x OUTPUT_SIZE] of length SEQ_LENGTH
        output_logits_BxO_S = []

        # Check if we want to collect cell history for the visualization purposes.
        if AppState().visualize:
            self.cell_state_history = []
            
        # Divide sequence into chunks of size [BATCH_SIZE x INPUT_SIZE] and process them one by one.
        for input_t_Bx1xI in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            # Process one chunk.
            output_BxO,  cell_state = self.ntm_cell(input_t_Bx1xI.squeeze(1), cell_state)
            # Append to list of logits.
            output_logits_BxO_S += [output_BxO]

        # Stack logits along time axis (1).
        output_logits_BxSxO = torch.stack(output_logits_BxO_S, 1)

        # Collect history for the visualization purposes.
        if AppState().visualize:
            self.cell_state_history.append(cell_state)

        return output_logits_BxSxO


    def plot_sequence(self, input_seq, output_seq, target_seq):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        
        # Change fonts globally - for all figures at once.
        rc('font',**{'family':'Times New Roman'})
        
        # Change to np arrays and transpose, so x will be time axis.
        input_seq = input_seq.numpy()
        output_seq = output_seq.numpy()
        target_seq = target_seq.numpy()

        x = np.transpose(np.zeros(input_seq.shape))
        y = np.transpose(np.zeros(output_seq.shape))
        z = np.transpose(np.zeros(target_seq.shape))
        
        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))
        # List of figures.
        figs = []
        for i, (input_word, output_word, target_word) in enumerate(zip(input_seq, output_seq, target_seq)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0]//10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))
            fig = Figure()
            axes = fig.subplots(3, 1, sharex=True, sharey=False,
                                gridspec_kw={'width_ratios': [input_seq.shape[0]]})

            # Set ticks.
            axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Set labels.
            axes[0].set_title('Inputs', fontname='Times New Roman', fontsize=13) 
            axes[0].set_ylabel('Control/Data bits')     
            axes[1].set_title('Targets')
            axes[1].set_ylabel('Data bits')
            axes[2].set_title('Predictions')
            axes[2].set_ylabel('Data bits')
            axes[2].set_xlabel('Item number')

            # Add words to adequate positions.
            x[:, i] = input_word
            y[:, i] = target_word
            z[:, i] = output_word
            # "Show" data on "axes".
            axes[0].imshow(x, interpolation='nearest', aspect='auto')
            axes[1].imshow(y, interpolation='nearest', aspect='auto')
            axes[2].imshow(z, interpolation='nearest', aspect='auto')
            # Append figure to a list.
            fig.set_tight_layout(True)
            figs.append(fig)

        # Set figure list to plot.
        self.plot.update(figs)
        return self.plot.is_closed




if __name__ == "__main__":
    # Set logging level.
    logging.basicConfig(level=logging.DEBUG)
    # "Loaded parameters".
    params = {'num_control_bits': 2, 'num_data_bits': 8, # input and output size
        'controller': {'name': 'rnn', 'hidden_state_size': 5,  'num_layers': 1, 'non_linearity': 'none'},  # controller parameters
        'interface': {'num_read_heads': 2,  'shift_size': 3},  # interface parameters
        'memory': {'num_addresses' :4, 'num_content_bits': 7} # memory parameters
        }
        
    logger = logging.getLogger('NTM-Module')
    logger.debug("params: {}".format(params))    
        
    input_size = params["num_control_bits"] + params["num_data_bits"]
    output_size = params["num_data_bits"]
        
    seq_length = 1
    batch_size = 2
    
    # Construct our model by instantiating the class defined above
    model = NTM(params)
    
    # Check for different seq_lengts and batch_sizes.
    for i in range(2):
        # Create random Tensors to hold inputs and outputs
        x = torch.randn(batch_size, seq_length,   input_size)
        y = torch.randn(batch_size, seq_length,  output_size)

        # Test forward pass.
        logger.info("------- forward -------")
        y_pred = model(x)

        logger.info("------- result -------")
        logger.info("input {}:\n {}".format(x.size(), x))
        logger.info("target.size():\n {}".format(y.size()))
        logger.info("prediction {}:\n {}".format(y_pred.size(), y_pred))
    
        # Change batch size and seq_length.
        seq_length = seq_length+1
        batch_size = batch_size+1
    
