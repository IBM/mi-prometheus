
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

# Add path to main project directory - so we can test the base plot, saving images, movies etc.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),  '..', '..')) 

from models.sequential_model import SequentialModel
from models.dnc.dnc_cell import DNCCell
from misc.app_state import AppState


class DNC(SequentialModel):
    """ @Ryan CLASS DESCRIPTION HERE """

    def __init__(self, params):
        """Initialize an DNC Layer.

        :param params: dictionary of inputs.
        """
        # Call base class initialization.
        super(DNC, self).__init__(params)

        try:
            self.output_units  = params['output_bits']
        except KeyError:
            self.output_units = params['data_bits']

        self.memory_addresses_size = params["memory_addresses_size"]
        self.label = params["name"]
        self.cell_state_history = None 

        # Create the DNC components
        self.DNCCell = DNCCell(self.output_units,params)

    def forward(self, data_tuple):       # inputs : batch_size, seq_len, input_size
        """
        Runs the DNC cell and plots if necessary
        
        :param data_tuple: Tuple containing inputs and targets
        :returns: output [batch_size, seq_len, output_size]
        """

        (inputs, targets) = data_tuple

        dtype = AppState().dtype

        output = None

        if self.app_state.visualize:
            self.cell_state_history = []

        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
       
        memory_addresses_size = self.memory_addresses_size

        #if memory size is not fixed, set it to the total input plus output size
        if memory_addresses_size == -1:
            memory_addresses_size = seq_length  

        # init state
        cell_state = self.DNCCell.init_state(memory_addresses_size,batch_size)


        #cell_state = self.init_state(memory_addresses_size)
        for j in range(seq_length):
            output_cell, cell_state = self.DNCCell(inputs[..., j, :], cell_state)

            if output_cell is None:
                continue

            output_cell = output_cell[..., None, :]
            if output is None:
                output = output_cell
            else:
                output = torch.cat([output, output_cell], dim=-2)

            #if self.plot_active:
            #    self.plot_memory_attention(output, cell_state)

        return output

    def plot_memory_attention(self, data_tuple, predictions, sample_number = 0):
        """
        Plots memory and attention TODO: fix

        :param data_tuple: Data tuple containing input [BATCH_SIZE x SEQUENCE_LENGTH x INPUT_DATA_SIZE] and target sequences  [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param predictions: Prediction sequence [BATCH_SIZE x SEQUENCE_LENGTH x OUTPUT_DATA_SIZE]
        :param sample_number: Number of sample in batch (DEFAULT: 0) 
        """        
        # plot attention/memory

        from models.dnc.plot_data import plot_memory_attention
        #plot_memory_attention(output, states[2], states[1][0], states[1][1], states[1][2], self.label)


