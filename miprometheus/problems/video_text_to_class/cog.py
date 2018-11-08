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
import gzip
import json
import os

from miprometheus.utils.data_dict import DataDict
from miprometheus.problems.video_text_to_class.video_text_to_class_problem import VideoTextToClassProblem

class COGDataset(VideoTextToClassProblem):
	"""
	The COG dataset is a sequential VQA dataset. Inputs are a sequence of images of simple shapes and characters on a black \
 	background, and a question based on these objects that relies on memory which has to be answered at every step of the \
	sequence.

 	..warning::

		Currently only implementing the subset of tasks that are classification. There are also regression tasks.

	"""

	def __init__(self, params):
		"""
		Initializes the COG Dataset problem:

			- Calls ``problems.problem.VideoTextToClassProblem`` class constructor,
			- Sets the following attributes using the provided ``params``:

				- ``self.root_folder`` (`string`) : Root directory of dataset where ``processed/training.pt``\
					and ``processed/test.pt`` will be saved,
				- ``self.data_folder`` (`string`) : Data directory where dataset is stored.
				- ``self.set`` (`string`) : 'val', 'test', or 'train'
				- ``self.tasks`` (`list of string`): Tasks to include, to implement later.
				- ``self.dataset_type`` (`string`) : Which dataset to use, 'canonical' or 'hard'. Will add \
								'generate' with options later.
				- ``self.tasks`` (`string or list of string`) : Which tasks to use. 'class', 'reg', 'all', or a 
\ list of tasks such as ['AndCompareColor', 'AndCompareShape']. Only selected tasks will be used.

		"""
	
		# Call base class constructors
		super(COGDataset, self).__init__(params)

		# Retrieve parameters from the dictionary
		self.root_folder= params['root_folder']
		self.data_folder= params['data_folder']
		self.set	= params['set']
		assert self.set in ['val','test','train'], "set in configuration file must be one of 'val', 'test', or 'train', "\
								"got {}".format(self.set)
		self.dataset_type	= params['dataset_type']
		assert self.dataset_type in ['canonical','hard','generate'], "dataset in configuration file must be one of "\
								"'canonical', 'hard', or 'generate', got {}".format(self.dataset_type)

		# Name
		self.name = 'COGDataset'

		self.tasks = params['tasks']
		if tasks == 'class':
			self.tasks = ['AndCompareColor','AndCompareShape','AndSimpleCompareColor',
'AndSimpleCompareShape','CompareColor','CompareShape','Exist','ExistColor','ExistColorOf','ExistColorSpace','ExistLastColorSameShape',
'ExistLastObjectSameObject','ExistLastShapeSameColor','ExistShape','ExistShapeOf','ExistShapeSpace','ExistSpace','GetColor','GetColorSpace',
'GetShape','GetShapeSpace','SimpleCompareColor','SimpleCompareShape']
		elif tasks == 'reg':
			self.tasks = ['AndSimpleExistColorGo','AndSimpleExistGo','AndSimpleExistShapeGo','CompareColorGo','CompareShapeGo','ExistColorGo',
'ExistColorSpaceGo','ExistGo','ExistShapeGo','ExistShapeSpaceGo','ExistSpaceGo','Go','GoColor','GoColorOf','GoShape','GoShapeOf',
'SimpleCompareColorGo','SimpleCompareShapeGo','SimpleExistColorGo','SimpleExistGo','SimpleExistShapeGo']

		# Load up .json files and set data definitions
		folder_name_append = ' '
		if self.dataset_type == 'canonical':
			folder_name_append = '_4_3_1'
			self.data_definitions = {'images': {'size': [-1, 4, 3, 112, 112], 'type': [torch.Tensor]},
			'questions': 	 {'size': [-1, 4, 1], 'type': [list, str]},
			'targets' :	 {'size': [-1, 4, 1], 'type': [torch.Tensor]},
			'targets_label':{'size': [-1, 1], 'type' : [list,str]}
			}

		elif self.dataset_type == 'hard':
			folder_name_append = '_8_7_10'
			self.data_definitions = {'images': {'size': [-1, 8, 3, 112, 112], 'type': [torch.Tensor]},
			'questions': 	 {'size': [-1, 8, 1], 'type': [list, str]},
			'targets' :	 {'size': [-1, 8, 1], 'type': [torch.Tensor]},
			'targets_label':{'size': [-1, 1], 'type' : [list,str]}
			}

		self.data_folder_path = os.path.join(self.data_folder,'data'+folder_name_append,self.set+folder_name_append)

		assert os.path.isdir(self.data_folder_path), "Data directory not found at {}. Please download the dataset and "\
	"point to the correct directory.".format(self.data_folder_path)
		
		#self.tasklist = []
		self.dataset = {}
	
		for tasklist in os.listdir(self.data_folder_path):
			self.dataset[(tasklist[4:-8])]=[]
			with gzip.open(os.path.join(self.data_folder_path,tasklist)) as f:
				fulltask = f.read().decode('utf-8').split('\n')
				for datapoint in fulltask:
					self.dataset[tasklist[4:-8]].append(json.loads(datapoint))

	def __getitem__(self, index):
		"""
		Getter method to access the dataset and return a sample.

		:param index: index of the sample to return.
		:type index: int
		:return: ``DataDict({'images', 'questions', 'targets', 'targets_label'})``, with:
		
			-images:	Sequence of images,
			-questions:	Sequence of questions,
			-targets:	Sequence of targets,
			-targets_label:	Targets' label

		"""

		# get sample
		# probably do the conversion here to images from json.

if __name__ == "__main__":

	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	params.add_config_params({'data_folder': '/home/esevgen/IBM/cog-master', 'root_folder': ' ', 'set': 'val', 'dataset_type': 'canonical'})

	cog_dataset = COGDataset(params)


	#with gzip.open('/home/esevgen/IBM/cog-master/data_4_3_1/val_4_3_1/cog_AndCompareColor.json.gz', 'r') as f:
	#	json_bytes=f.read()
	#json_str = json_bytes.decode('utf-8').split('\n')
	#data = json.loads(json_str[0])
	#print(data)
	#print(data['objects'])
	#print(data['question'])

 
