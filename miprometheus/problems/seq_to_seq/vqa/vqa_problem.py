#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vqa_problem.py: abstract base class for sequential VQA problems."""

__author__ = "Emre Sevgen"

import torch
import torch.nn as nn
import numpy as np
from miprometheus.problems.seq_to_seq.seq_to_seq_problem import SeqToSeqProblem


class VQAProblem(SeqToSeqProblem):
	"""
	Abstract base class for sequential VQA problems.

	COG inherits from it (for now).

	Provides some basic features useful in all problems of such type.
	
	"""

	def __init__(self, params):
		# Should 'questions' be [-1, 1] or [-1, -1, 1], as in an entry for each member of a sequence?
		"""
		Initializes problem:
	
		- Calls :py:class:`miprometheus.problems.SeqToSeqProblem` class constructor,
		- Sets loss function to :py:class:`torch.nn.CrossEntropyLoss`,
		- Sets ``self.data_definitions`` to:

		>>> self.data_definitions = {'images': {'size': [-1, -1, 3, -1, -1], 'type': [torch.Tensor]},
		>>>                          'mask': {'size': [-1, -1, 1], 'type': [torch.Tensor]},
		>>>                          'questions' {'size': [-1, 1], 'type': [list, str]},
		>>>                          'targets': {'size': [-1, -1,  1], 'type': [torch.Tensor]},
		>>>                          'targets_label': {'size': [-1, 1], 'type': [list, str]}
		>>>                         }

		:param params: Dictionary of parameters (read from configuration ``.yaml`` file).
		:type params: :py:class:`miprometheus.utils.ParamInterface`

		"""
		super(VQAProblem, self).__init__(params)

		# Set default loss function to cross entropy.
		self.loss_function = nn.CrossEntropyLoss()

		# Set default data_definitions dict
		# Should 'questions' be [-1, 1] or [-1, -1, 1], as in an entry for each member of a sequence?
		self.data_definitions = {'images': {'size': [-1, -1, 3, -1, -1], 'type': [torch.Tensor]},
								 'mask': {'size': [-1, -1, 1], 'type': [torch.Tensor]},
								 'questions': {'size': [-1, -1, 1], 'type': [list, str]},
								 'targets': {'size': [-1, -1,  1], 'type': [torch.Tensor]},
								 'targets_label': {'size': [-1, 1], 'type': [list, str]}}

		# Default problem name.
		self.name = 'VQAProblem'

	def show_sample(self, data_dict, sample_number=0, sequence_number=0):
		"""
		Shows a sample from the batch.

		:param data_dict: ``DataDict`` containing inputs and targets.
		:type data_dict: :py:class:`miprometheus.utils.DataDict`

		:param sample_number: Number of sample in batch (default: 0)
		:type sample_number: int

		:param sequence_number: Which image in the sequence to display (default: 0)
		:type sequence_number: int

		"""
		import matplotlib.pyplot as plt

		# Unpack dict.
		images = data_dict['images']
		targets = data_dict['targets']
		labels = data_dict['targets_label']
		questions = data_dict['questions']

		# Get sample.
		images = images[sample_number].cpu().detach().numpy()
		targets = targets[sample_number].cpu().detach().numpy()
		labels = labels[sample_number]
		question = questions[sample_number]

		# Get image and label in sequence.
		image = images[sequence_number]
		target = targets[sequence_number]
		label = labels[sequence_number]

		# Reshape image.
		if image.shape[0] == 1:
			# This is a single channel image - get rid of this dimension
			image = np.squeeze(image, axis=0)
		else:
			# More channels - move channels to axis2, according to matplotilb documentation.
			# (X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4))
			image = image.transpose(1, 2, 0)

		# show data.
		plt.xlabel('num_columns')
		plt.ylabel('num_rows')
		plt.title('Target: {} ({}), {}th in Sequence, Question: {}'.format(label, target, sequence_number, question))
		plt.imshow(image, interpolation='nearest', aspect='auto')

		# Plot!
		plt.show()


if __name__ == '__main__':

	from miprometheus.utils.param_interface import ParamInterface
	
	sample = VQAProblem(ParamInterface())[0]
	
	print(repr(sample))
