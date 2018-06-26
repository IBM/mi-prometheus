import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from misc.app_state import AppState

def forward_step(model, problem, episode, stat_col, data_tuple,  aux_tuple):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # convert to CUDA
    if AppState().use_CUDA:
        data_tuple, aux_tuple = problem.turn_on_cuda(data_tuple, aux_tuple)

    # Perform forward calculation.
    logits = model(data_tuple)

    # Evaluate loss function.
    loss = problem.evaluate_loss(data_tuple, logits, aux_tuple)

    # Collect "elementary" statistics - episode and loss.
    stat_col['episode'] = episode
    stat_col['loss'] = loss

    # Collect other (potential) statistics from problem & model.
    problem.collect_statistics(stat_col, data_tuple, logits, aux_tuple)
    model.collect_statistics(stat_col, data_tuple, logits)

    # Return tuple: logits, loss.
    return logits, loss 

def check_and_set_cuda(params, logger):
    """ Enables Cuda if available and sets the default data types
    :param params: paramater interface object containing either training or testing parameters
    :param logger: logger object
    """
    turn_on_cuda = False
    try:  # If the 'cuda' key is not present, catch the exception and do nothing
        turn_on_cuda = params['cuda']
    except KeyError:
        pass


    # Determine if CUDA is to be used.
    if torch.cuda.is_available():
        if turn_on_cuda:
            AppState().convert_cuda_types()
            logger.info('Running with CUDA enabled')
    elif turn_on_cuda:
        logger.warning('CUDA is enabled but there is no available device')

    #TODO Add flags to change these
    AppState().set_dtype('float')
    AppState().set_itype('int')
