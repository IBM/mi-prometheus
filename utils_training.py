import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def forward_step(model, problem, episode, stat_col, data_tuple,  aux_tuple,  use_CUDA):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # convert to CUDA
    if use_CUDA:
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

