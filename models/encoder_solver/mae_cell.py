#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch
import collections

# Set logging level.
import logging
logger = logging.getLogger('MAE-Cell')
# logging.basicConfig(level=logging.DEBUG)

import os

from models.controllers.controller_factory import ControllerFactory


from models.encoder_solver.mae_interface import MAEInterface

# Helper collection type.
_MAECellStateTuple = collections.namedtuple(
    'MAECellStateTuple', ('ctrl_state', 'interface_state', 'memory_state'))


class MAECellStateTuple(_MAECellStateTuple):
    """
    Tuple used by MAE Cells for storing current/past state information.
    """
    __slots__ = ()


class MAECell(torch.nn.Module):
    """
    Class representing a single Memory-Augmented Encoder cell.
    """

    def __init__(self, params):
        """
        Cell constructor. Cell creates controller and interface. It also
        initializes memory "block" that will be passed between states.

        :param params: Dictionary of parameters.

        """
        # Call constructor of base class.
        super(MAECell, self).__init__()

        # Parse parameters.
        # Set input and output sizes.
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        try:
            self.output_size = params['num_output_bits']
        except KeyError:
            self.output_size = params['num_data_bits']

        # Get controller hidden state size.
        self.controller_hidden_state_size = params['controller'][
            'hidden_state_size']

        # Controller - entity that processes input and produces hidden state of
        # the ntm cell.
        ext_controller_inputs_size = self.input_size
        # Create dictionary wirh controller parameters.
        controller_params = {
            "name": params['controller']['name'],
            "input_size": ext_controller_inputs_size,
            "output_size": self.controller_hidden_state_size,
            "non_linearity": params['controller']['non_linearity'],
            "num_layers": params['controller']['num_layers']
        }
        # Build the controller.
        self.controller = ControllerFactory.build_model(controller_params)

        # Interface - entity responsible for accessing the memory.
        self.interface = MAEInterface(params)

        # Layer that produces output on the basis of... hidden state?
        ext_hidden_size = self.controller_hidden_state_size
        self.hidden2output = torch.nn.Linear(ext_hidden_size, self.output_size)

    def load(self, filename):
        # Check filename.
        if os.path.isfile(filename):
            # Load checkpoint from filename.
            chkpt = torch.load(
                filename, map_location=lambda storage, loc: storage)
            # Load controller and interface
            self.controller.load_state_dict(chkpt['ctrl_dict'])
            self.interface.load_state_dict(chkpt['interface_dict'])
            logger.info(
                "Imported {} parameters from checkpoint (episode {}, loss {}) from file {}".format(
                    chkpt['name'],
                    chkpt['stats']['episode'],
                    chkpt['stats']['loss'],
                    filename))
        else:
            logger.error("Encoder checkpoint not found at {}".format(filename))

    def save(self, model_dir, stat_col, is_best_model, save_intermediate):
        """
        Method saves the model and encoder to file.

        :param model_dir: Directory where the model will be saved.
        :param stat_col: Statistics collector that contain current loss and episode number (and other statistics).
        :param is_best_model: Flag indicating whether it is the best model or not.
        :parma save_intermediate: Flag indicating whether intermediate models should be saved or not.

        """
        episode = stat_col['episode']
        # Checkpoint to be saved.
        chkpt = {
            'name': 'MAE controller and interface',
            'ctrl_dict': self.controller.state_dict(),
            'interface_dict': self.interface.state_dict(),
            'stats': stat_col.statistics
        }

        # Save the intermediate checkpoint.
        if save_intermediate:
            # Generate filename pt.
            filename = model_dir + 'encoder_episode_{:05d}.pt'.format(episode)
            # Save dictionary to file.
            torch.save(chkpt, filename)
            logger.info(
                "Encoder and statistics exported to checkpoint {}".format(
                    filename))

        # Save the best model.
        if is_best_model:
            # Generate filename pt.
            filename = model_dir + 'encoder_best.pt'
            # Save dictionary to file.
            torch.save(chkpt, filename)
            logger.info(
                "Encoder and statistics exported to checkpoint {}".format(
                    filename))

    def freeze(self):
        """
        Freezes the trainable weigths.
        """
        # Freeze controller.
        for param in self.controller.parameters():
            param.requires_grad = False
        logger.info("Encoder controller is frozen")

        # Freeze interface.
        self.interface.freeze()
        logger.info("Encoder interface is frozen")

        # Freeze output layer.
        # for param in self.hidden2output.parameters():
        #    param.requires_grad = False

    def init_state(self, init_memory_BxAxC):
        """
        Initializes state of MAE cell. Recursively initialization: controller,
        interface.

        :param init_memory_BxAxC: Initial memory state [BATCH_SIZE x MEMORY_ADDRESSES x MEMORY_CONTENT].
        :returns: Initial state tuple - object of NTMCellStateTuple class.

        """
        # Get number of memory addresses.
        batch_size = init_memory_BxAxC.size(0)
        num_memory_addresses = init_memory_BxAxC.size(1)

        # Initialize controller state.
        ctrl_init_state = self.controller.init_state(batch_size)

        # Initialize interface state.
        interface_init_state = self.interface.init_state(
            batch_size, num_memory_addresses)

        # Pack and return a tuple.
        return MAECellStateTuple(
            ctrl_init_state, interface_init_state, init_memory_BxAxC)

    def forward(self, inputs_BxI, prev_cell_state):
        """
        Forward function of NTM cell.

        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a MAECellStateTuple tuple, containing previous state of the cell.
        :returns: MAECellStateTuple tuple containing current cell state.

        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,
         prev_memory_BxAxC) = prev_cell_state

        controller_input = inputs_BxI
        # Execute controller forward step.
        ctrl_output_BxH, ctrl_state_tuple = self.controller(
            controller_input, prev_ctrl_state_tuple)

        # Execute interface forward step.
        memory_BxAxC, interface_state_tuple = self.interface(
            ctrl_output_BxH, prev_memory_BxAxC, prev_interface_state_tuple)

        # Output layer - takes controller hidden state.
        logits_BxO = self.hidden2output(ctrl_output_BxH)

        # Pack current cell state.
        cell_state_tuple = MAECellStateTuple(
            ctrl_state_tuple, interface_state_tuple, memory_BxAxC)

        # Return logits and current cell state.
        return logits_BxO, cell_state_tuple
