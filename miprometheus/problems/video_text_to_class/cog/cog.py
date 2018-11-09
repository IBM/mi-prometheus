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
from miprometheus.problems.video_text_to_class.cog import train_utils as tu

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
		if self.tasks == 'class':
			self.tasks = ['AndCompareColor','AndCompareShape','AndSimpleCompareColor','AndSimpleCompareShape','CompareColor','CompareShape','Exist',
'ExistColor','ExistColorOf','ExistColorSpace','ExistLastColorSameShape','ExistLastObjectSameObject','ExistLastShapeSameColor',
'ExistShape','ExistShapeOf','ExistShapeSpace','ExistSpace','GetColor','GetColorSpace','GetShape','GetShapeSpace','SimpleCompareColor',
'SimpleCompareShape']
		elif self.tasks == 'reg':
			self.tasks = ['AndSimpleExistColorGo','AndSimpleExistGo','AndSimpleExistShapeGo','CompareColorGo','CompareShapeGo','ExistColorGo',
'ExistColorSpaceGo','ExistGo','ExistShapeGo','ExistShapeSpaceGo','ExistSpaceGo','Go','GoColor','GoColorOf','GoShape','GoShapeOf',
'SimpleCompareColorGo','SimpleCompareShapeGo','SimpleExistColorGo','SimpleExistGo','SimpleExistShapeGo']
		elif self.tasks == 'all':
			self.tasks =['AndCompareColor','AndCompareShape','AndSimpleCompareColor','AndSimpleCompareShape','CompareColor','CompareShape','Exist',
'ExistColor','ExistColorOf','ExistColorSpace','ExistLastColorSameShape','ExistLastObjectSameObject','ExistLastShapeSameColor',
'ExistShape','ExistShapeOf','ExistShapeSpace','ExistSpace','GetColor','GetColorSpace','GetShape','GetShapeSpace','SimpleCompareColor',
'SimpleCompareShape','AndSimpleExistColorGo','AndSimpleExistGo','AndSimpleExistShapeGo','CompareColorGo','CompareShapeGo','ExistColorGo',
'ExistColorSpaceGo','ExistGo','ExistShapeGo','ExistShapeSpaceGo','ExistSpaceGo','Go','GoColor','GoColorOf','GoShape','GoShapeOf',
'SimpleCompareColorGo','SimpleCompareShapeGo','SimpleExistColorGo','SimpleExistGo','SimpleExistShapeGo']

		# Load up .json files and set data definitions
		folder_name_append = ' '
		
		if self.dataset_type == 'canonical':
			folder_name_append = '_4_3_1'			
			self.sequence_length = 4
		elif self.dataset_type == 'hard':
			folder_name_append = '_8_7_10'			
			self.sequence_length = 8
			
		self.data_definitions = {'images': {'size': [-1, self.sequence_length, 3, 112, 112], 'type': [torch.Tensor]},
					'questions': 	 {'size': [-1, 1], 'type': [list, str]},
					'targets' :	 {'size': [-1, self.sequence_length, 2], 'type': [torch.Tensor]},
					'targets_label':{'size': [-1, 1], 'type' : [list,str]}
					}

		self.data_folder_path = os.path.join(self.data_folder,'data'+folder_name_append,self.set+folder_name_append)

		assert os.path.isdir(self.data_folder_path), "Data directory not found at {}. Please download the dataset and "\
	"point to the correct directory.".format(self.data_folder_path)
		
		#self.tasklist = []
		self.dataset = {}
		self.length = 0
	
		for tasklist in os.listdir(self.data_folder_path):
			if tasklist[4:-8] in self.tasks:
				self.dataset[(tasklist[4:-8])]=[]
				with gzip.open(os.path.join(self.data_folder_path,tasklist)) as f:
					fulltask = f.read().decode('utf-8').split('\n')
					for datapoint in fulltask:
						self.dataset[tasklist[4:-8]].append(json.loads(datapoint))
				self.length = self.length + len(self.dataset[tasklist[4:-8]])

		#print(tu.json_to_feeds(self.dataset['AndCompareShape']))

	def __getitem__(self, index):
		"""
		Getter method to access the dataset and return a sample.

		:param index: index of the sample to return.
		:type index: int
		:return: ``DataDict({'images', 'questions', 'targets', 'targets_label'})``, with:
		
			-images:	Sequence of images,
			-questions:	Question on the sequence (this is constant per sequence for COG),
			-targets:	Sequence of targets,
			-targets_label:	Targets' label

		"""
		# With the assumption that each family has the same number of examples
		i = index % len(self.tasks)
		j = int(index / len(self.tasks))

		# This returns:
		# All variables are numpy array of float32
			# in_imgs: (n_epoch*batch_size, img_size, img_size, 3)
			# in_rule: (max_seq_length, batch_size) the rule language input, type int32
			# seq_length: (batch_size,) the length of each task instruction
			# out_pnt: (n_epoch*batch_size, n_out_pnt)
			# out_pnt_xy: (n_epoch*batch_size, 2)
			# out_word: (n_epoch*batch_size, n_out_word)
			# mask_pnt: (n_epoch*batch_size)
			# mask_word: (n_epoch*batch_size)		
		output = tu.json_to_feeds([self.dataset[self.tasks[i]][j]])[0]
		images = ((torch.from_numpy(output)).permute(1,0,4,2,3)).squeeze()
				

		data_dict = DataDict({key: None for key in self.data_definitions.keys()})
		data_dict['images']	= images
		data_dict['questions']	= self.dataset[self.tasks[i]][j]['question']
		data_dict['targets']	= self.dataset[self.tasks[i]][j]['answers']
		data_dict['targets_label'] = self.dataset[self.tasks[i]][j]['answers']
		

		return(data_dict)

	def collate_fn(self, batch):
		"""
		Combines a list of ``DataDict`` (retrieved with ``__getitem__``) into a batch.

		:param batch: list of individual ``DataDict`` samples to combine.
		:return: ``DataDict({'images', 'questions', 'targets', 'targets_label'})`` containing the batch.
		"""
		return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(COGDataset, self).collate_fn(batch).values())})
		

if __name__ == "__main__":

	# Define useful params
	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	params.add_config_params({'data_folder': '/home/esevgen/IBM/cog-master', 'root_folder': ' ', 'set': 'val', 'dataset_type': 'canonical','tasks': 'all'})

	# Create problem
	cog_dataset = COGDataset(params)

	# Set batch size such that there's one input of each task type.
	batch_size = 44

	# Get a sample
	sample = cog_dataset[0]
	print(repr(sample))
	print('__getitem__ works')
	
	# Set up Dataloader iterator
	from torch.utils.data import DataLoader

	dataloader = DataLoader(dataset=cog_dataset, collate_fn=cog_dataset.collate_fn,
		            batch_size=batch_size, shuffle=True, num_workers=8)

	# Display single sample (0) from batch.
	batch = next(iter(dataloader))

	cog_dataset.show_sample(batch,0,0)	



	#with gzip.open('/home/esevgen/IBM/cog-master/data_4_3_1/val_4_3_1/cog_AndCompareColor.json.gz', 'r') as f:
	#	json_bytes=f.read()
	#json_str = json_bytes.decode('utf-8').split('\n')
	#data = json.loads(json_str[0])
	#print(data)
	#print(data['objects'])
	#print(data['question'])

 
