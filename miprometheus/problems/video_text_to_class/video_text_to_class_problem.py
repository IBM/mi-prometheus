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

"""video_text_to_class_problem.py: abstract base class for sequential VQA problems."""

import torch
import torch.nn as nn
import numpy as np
from miprometheus.problems.problem import Problem


class VideoTextToClassProblem(Problem):
	"""
	Abstract base class for sequential VQA problems.

	COG inherits from it (for now).

	Provides some basic features useful in all problems of such type.
	
	"""

	def __init__(self, params):
		# Should 'questions' be [-1, 1] or [-1, -1, 1], as in an entry for each member of a sequence?
		"""
		Initializes problem:
	
		- Calls ``problems.problem.Problem`` class constructor,
		- Sets loss function to ``CrossEntropy``,
		- Sets ``self.data_definitions`` to:

		>>> self.data_definitions = {'images': {'size': [-1, -1, 3, -1, -1], 'type': [torch.Tensor]},
		>>>			     'mask': {'size': [-1, -1, 1], 'type': [torch.Tensor]},
		>>>			     'questions' {'size': [-1, 1], 'type': [list, str]},	
		>>>			     'targets': {'size': [-1, -1,  1], 'type': [torch.Tensor]},
		>>>			     'targets_label': {'size': [-1, 1], 'type': [list, str]}
		>>>			    }

		:param params: Dictionary of parameters (read from configuration ``.yaml`` file).
		"""
		super(VideoTextToClassProblem, self).__init__(params)

		# Set default loss function to cross entropy.
		self.loss_function = nn.CrossEntropyLoss()

		# Set default data_definitions dict
		# Should 'questions' be [-1, 1] or [-1, -1, 1], as in an entry for each member of a sequence?
		self.data_definitions = {'images': {'size': [-1, -1, 3, -1, -1], 'type': [torch.Tensor]},
					'mask': {'size': [-1, -1, 1], 'type': [torch.Tensor]},
					'questions': {'size': [-1, -1, 1], 'type': [list, str]},	
					'targets': {'size': [-1, -1,  1], 'type': [torch.Tensor]},
					'targets_label': {'size': [-1, 1], 'type': [list, str]}
					}

		# Default problem name.
		self.name = 'VideoTextToClassProblem'

	def show_sample(self, data_dict, sample_number=0):
		"""
		Shows a sample from the batch.

		:param data_dict: ``DataDict`` containing inputs and targets.
		:type data_dict: DataDict

		:param sample_number: Number of sample in batch (default: 0)
		:type sample_number: int

		"""
		import matplotlib.pyplot as plt

		images = data_dict['images']
		target = data_dict['targets']
		label = data_dict['targets_label']

		# Get sample.
		image = images[sample_number].cpu().detach().numpy()
		target = targets[sample_number].cpu().detach().numpy()
		label = labels[sample_number]

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
		plt.title('Target class: {} ({})'.format(label, target))
		plt.imshow(image, interpolation='nearest', aspect='auto', cmap='gray_r')

		# Plot!
		plt.show()

if __name__ == '__main__':

	from miprometheus.utils.param_interface import ParamInterface
	
	sample = VideoTextToClassProblem(ParamInterface())[0]
	
	print(repr(sample))
