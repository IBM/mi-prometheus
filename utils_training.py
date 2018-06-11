import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def forward_step(model, problem, data_tuple,  aux_tuple,  use_CUDA):
    """ Function performs a single forward step.

    :returns: logits, loss and accuracy (former using provided criterion)
    """
    # convert to CUDA
    if use_CUDA:
        data_tuple, aux_tuple = problem.turn_on_cuda(data_tuple, aux_tuple)

    # Perform forward calculation.
    logits = model(data_tuple)

    loss, accuracy = problem.evaluate_loss_accuracy(logits, data_tuple, aux_tuple)
    # Return tuple: logits, loss, accuracy.
    return logits, loss, accuracy

