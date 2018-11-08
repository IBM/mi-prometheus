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

if __name__ == '__main__':

	from miprometheus.utils.param_interface import ParamInterface
	
	sample = VideoTextToClassProblem(ParamInterface())[0]
	
	print(repr(sample))
