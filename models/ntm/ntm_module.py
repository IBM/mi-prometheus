#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_module.py: pytorch module implementing Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import logging
import numpy as np
import io
import pickle

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
        if self.num_memory_addresses == -1:
            # Set equal to input sequence length.
            self.num_memory_addresses = inputs_BxSxI.size(1)
        
        # Initialize 'zero' state.
        cell_state = self.ntm_cell.init_state(inputs_BxSxI.size(0),  self.num_memory_addresses)

        # List of output logits [BATCH_SIZE x OUTPUT_SIZE] of length SEQ_LENGTH
        output_logits_BxO_S = []

        # Check if we want to collect cell history for the visualization purposes.
        if AppState().visualize:
            self.cell_state_history = []
            self.cell_state_initial = cell_state
            
        # Divide sequence into chunks of size [BATCH_SIZE x INPUT_SIZE] and process them one by one.
        for input_t_Bx1xI in inputs_BxSxI.chunk(inputs_BxSxI.size(1), dim=1):
            # Process one chunk.
            output_BxO,  cell_state = self.ntm_cell(input_t_Bx1xI.squeeze(1), cell_state)
            # Append to list of logits.
            output_logits_BxO_S += [output_BxO]

            # Collect cell history - for the visualization purposes.
            if AppState().visualize:
                self.cell_state_history.append(cell_state)
                
        # Stack logits along time axis (1).
        output_logits_BxSxO = torch.stack(output_logits_BxO_S, 1)

        return output_logits_BxSxO

    def pickle_figure_template(self):
        from matplotlib.figure import Figure
        import matplotlib.ticker as ticker
        from matplotlib import rc
        import matplotlib.gridspec as gridspec
        
        # Change fonts globally - for all figures/subsplots at once.
        rc('font',**{'family':'Times New Roman'})
        
        # Prepare "generic figure template".
        # Create figure object.
        fig = Figure()
        #axes = fig.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'width_ratios': [input_seq.shape[0]]})
        
        # Create a specific grid for NTM .
        gs = gridspec.GridSpec(3, 7)

        # Memory
        ax_memory = fig.add_subplot(gs[:, 0]) # all rows, col 0
        ax_write_attention = fig.add_subplot(gs[:, 1:3]) # all rows, col 2-3
        ax_read_attention = fig.add_subplot(gs[:, 3:5]) # all rows, col 4-5

        ax_inputs = fig.add_subplot(gs[0, 5:]) # row 0, span 2 columns
        ax_targets = fig.add_subplot(gs[1, 5:]) # row 0, span 2 columns
        ax_predictions = fig.add_subplot(gs[2, 5:]) # row 0, span 2 columns
        
        # Set ticks - for bit axes only (for now).
        ax_inputs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_inputs.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_targets.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_targets.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_predictions.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_predictions.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_memory.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_memory.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_write_attention.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_write_attention.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_read_attention.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_read_attention.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Set labels.
        ax_inputs.set_title('Inputs') 
        ax_inputs.set_ylabel('Control/Data bits')     
        ax_targets.set_title('Targets')
        ax_targets.set_ylabel('Data bits')
        ax_predictions.set_title('Predictions')
        ax_predictions.set_ylabel('Data bits')
        ax_predictions.set_xlabel('Item number/Iteration')

        ax_memory.set_title('Memory') 
        ax_memory.set_ylabel('Memory Addresses')
        ax_memory.set_xlabel('Content bits')
        ax_write_attention.set_title('Write Attention') 
        ax_write_attention.set_xlabel('Iteration')      
        ax_read_attention.set_title('Read Attention') 
        ax_read_attention.set_xlabel('Iteration')
        
        # Create buffer and pickle the figure.
        buf = io.BytesIO()
        pickle.dump(fig, buf)
    
        return buf

    def plot_sequence(self, input_seq, output_seq, target_seq):
        """ Creates a default interactive visualization, with a slider enabling to move forth and back along the time axis (iteration in a given episode).
        The default visualizatoin contains input, output and target sequences.
        For more model/problem dependent visualization please overwrite this method in the derived model class.
        """
        #import time
        #start_time = time.time()
        # Create figure template.
        buf = self.pickle_figure_template()
        
        # Set intial values of displayed  inputs, targets and predictions - simply zeros.
        inputs_displayed = np.transpose(np.zeros(input_seq.shape))
        targets_displayed = np.transpose(np.zeros(target_seq.shape))
        predictions_displayed = np.transpose(np.zeros(output_seq.shape))
        
        # Set initial values of memory and attentions.
        # Unpack initial state.
        (ctrl_state,  interface_state,  memory_state,  read_vectors) = self.cell_state_initial 
        (read_attentions,  write_attention) = interface_state
        
        memory_displayed = memory_state[0]
        read0_attention_displayed = np.zeros((read_attentions[0].shape[1],  target_seq.shape[0]))
        write_attention_displayed = np.zeros((write_attention.shape[1],  target_seq.shape[0]))
       
        # Log sequence length - so the user can understand what is going on.
        logger = logging.getLogger('ModelBase')
        logger.info("Generating dynamic visualization of {} figures, please wait...".format(input_seq.shape[0]))
        # List of figures.
        figs = []
        for i, (input_element, output_elementd, target_element,  cell_state) in enumerate(zip(input_seq, output_seq, target_seq,  self.cell_state_history)):
            # Display information every 10% of figures.
            if (input_seq.shape[0] > 10) and (i % (input_seq.shape[0]//10) == 0):
                logger.info("Generating figure {}/{}".format(i, input_seq.shape[0]))
                 
           # Create figure object on the basis of template.
            buf.seek(0)
            fig = pickle.load(buf) 
            (ax_memory,  ax_read_attention,  ax_write_attention,  ax_inputs,  ax_targets,  ax_predictions) = fig.axes
 
            # Update displayed values on adequate positions.
            inputs_displayed[:, i] = input_element
            targets_displayed[:, i] = target_element
            predictions_displayed[:, i] = output_elementd

            # Unpack cell state.
            (ctrl_state,  interface_state,  memory_state,  read_vectors) = cell_state
            (read_attentions,  write_attention) = interface_state
            
            memory_displayed = memory_state[0].detach().numpy()
            # Get attention of head 0.
            read0_attention_displayed[:, i] = read_attentions[0][0][:, 0].detach().numpy()
            write_attention_displayed[:, i] = write_attention[0][:, 0].detach().numpy()

            # "Show" data on "axes".
            ax_memory.imshow(memory_displayed, interpolation='nearest', aspect='auto')
            ax_read_attention.imshow(read0_attention_displayed, interpolation='nearest', aspect='auto')
            ax_write_attention.imshow(write_attention_displayed, interpolation='nearest', aspect='auto')
            ax_inputs.imshow(inputs_displayed, interpolation='nearest', aspect='auto')
            ax_targets.imshow(targets_displayed, interpolation='nearest', aspect='auto')
            ax_predictions.imshow(predictions_displayed, interpolation='nearest', aspect='auto')
            
            # Append figure to a list.
            fig.set_tight_layout(True)
            figs.append(fig)

        #print("--- %s seconds ---" % (time.time() - start_time))
        # Update time plot fir generated list of figures.
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
    
