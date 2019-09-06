 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

"""cog.py: Implementation of Google's COG dataset. https://arxiv.org/abs/1803.06092"""

__author__ = "Emre Sevgen, Tomasz Kornuta, Vincent Albouy"

import torch
import torch.nn as nn
import gzip
import json
import os
import tarfile
import string
import numpy as np
from miprometheus.problems.seq_to_seq.video_text_to_class.video_text_to_class_problem import VideoTextToClassProblem
from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils import json_to_img as jti


class COG(VideoTextToClassProblem):
	"""
	The COG dataset is a sequential VQA dataset.

	Inputs are a sequence of images of simple shapes and characters on a black background, \
	and a question based on these objects that relies on memory which has to be answered at every step of the sequence.

	See https://arxiv.org/abs/1803.06092 (`A Dataset and Architecture for Visual Reasoning with a Working Memory`)\
	for the reference paper.

	"""

	def __init__(self, params):
		"""
		Initializes the :py:class:`COG` problem:

			- Calls :py:class:`miprometheus.problems.VQAProblem` class constructor,
			- Sets the following attributes using the provided ``params``:

				- ``self.data_folder`` (`string`) : Data directory where the dataset is stored.
				- ``self.set`` (`string`) : 'val', 'test', or 'train'
				- ``self.tasks`` (`string` or list of `string`) : Which tasks to use. 'class', 'reg', \
				'all', 'binary', or a list of tasks such as ['AndCompareColor', 'AndCompareShape']. \
				Only the selected tasks will be used.

				Classification tasks are: ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor',
				'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist',
				'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape',
				'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 'ExistShape',
				'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'GetColor', 'GetColorSpace',
				'GetShape', 'GetShapeSpace', 'SimpleCompareColor', 'SimpleCompareShape']		

				Regression tasks are: 		self.regression_tasks = ['AndSimpleExistColorGo', 'AndSimpleExistGo', 'AndSimpleExistShapeGo', 'CompareColorGo',
				'CompareShapeGo', 'ExistColorGo', 'ExistColorSpaceGo', 'ExistGo', 'ExistShapeGo',
				'ExistShapeSpaceGo', 'ExistSpaceGo', 'Go', 'GoColor', 'GoColorOf', 'GoShape',
				'GoShapeOf', 'SimpleCompareColorGo', 'SimpleCompareShapeGo', 'SimpleExistColorGo',
				'SimpleExistGo','SimpleExistShapeGo']

				Binary classification tasks are: ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor', 'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist', 
				'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape', 'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 
				'ExistShape', 'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'SimpleCompareColor', 'SimpleCompareShape'] 


				- ``self.dataset_type`` (`string`) : Which dataset to use, 'canonical', 'hard', or \
				'generated'. If 'generated', please specify 'examples_per_task', 'sequence_length', \
				'memory_length', and 'max_distractors' under 'generation'. Can also specify 'nr_processors' for generation.

			- Adds the following as default params:

				>>> {'data_folder': os.path.expanduser('~/data/cog'),
				>>>  'set': 'train',
				>>>  'tasks': 'class',
				>>>  'dataset_type': 'canonical',
				>>>  'initialization_only': False}

			- Sets:

				>>> self.data_definitions = {'images': {'size': [-1, self.sequence_length, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
				>>>                          'tasks': {'size': [-1, 1], 'type': [list, str]},
				>>>                          'questions': {'size': [-1, 1], 'type': [list, str]},
				>>>                          'targets_pointing': {'size': [-1, self.sequence_length, 2], 'type': [torch.Tensor]},
				>>>                          'targets_answer': {'size': [-1, self.sequence_length, 1], 'type' : [list,str]}
				>>>                         }



		:param params: Dictionary of parameters (read from configuration ``.yaml`` file).
		:type params: :py:class:`miprometheus.utils.ParamInterface`


		"""
	
		# Call base class constructors
		super(COG, self).__init__(params)

		# Set default parameters.
		self.params.add_default_params({'data_folder': os.path.expanduser('~/data/cog'), 'set': 'train',
										'tasks': 'class',
										'dataset_type': 'canonical',
										'initialization_only': False})

		# Retrieve parameters from the dictionary
		# Data folder main is /path/cog
		# Data folder parent is data_X_Y_Z
		# Data folder children are train_X_Y_Z, test_X_Y_Z, or val_X_Y_Z
		self.data_folder_main = os.path.expanduser(params['data_folder'])

		self.set = params['set']
		assert self.set in ['val', 'test', 'train'], "set in configuration file must be one of 'val', 'test', or " \
													 "'train', got {}".format(self.set)
		self.dataset_type = params['dataset_type']
		assert self.dataset_type in ['canonical', 'hard', 'generated'], "dataset in configuration file must be one of " \
																		"'canonical', 'hard', or 'generated', got {}".format(self.dataset_type)

		# Parse task and dataset_type
		self.parse_tasks_and_dataset_type(params)
	
		# Name
		self.name = 'COG'

		# Initialize word lookup dictionary
		self.word_lookup = {}

		# Initialize unique word counter. Updated by UpdateAndFetchLookup
		self.nr_unique_words = 1

		# This should be the length of the longest sentence encounterable
		self.nwords = 24

		# Get the "hardcoded" image width/height.
		self.img_size = 112  # self.params['img_size']

		self.output_classes_pointing=49

		# Set default values
		self.default_values = {	'height': self.img_size,
								'width': self.img_size,
								'num_channels': 3,
								'sequence_length' : self.sequence_length,
								'nb_classes': self.output_classes,
								'nb_classes_pointing': self.output_classes_pointing,
								'embed_vocab_size': self.input_words}
		
		# Set data dictionary based on parsed dataset type
		self.data_definitions = {
			'images': {'size': [-1, self.sequence_length, 3, self.img_size, self.img_size], 'type': [torch.Tensor]},
			'tasks':	{'size': [-1, 1], 'type': [list, str]},
			'questions': 	{'size': [-1,self.nwords], 'type': [torch.Tensor]},
			#'targets': {'size': [-1,self.sequence_length, self.output_classes], 'type': [torch.Tensor]},
			'targets_pointing' :	{'size': [-1, self.sequence_length, 2], 'type': [torch.Tensor]},
			'targets_answer':{'size': [-1, self.sequence_length, self.output_classes], 'type' : [list,str]},
			'masks_pnt':{'size': [-1, self.sequence_length ], 'type' : [torch.Tensor]},
			'masks_word':{'size': [-1, self.sequence_length], 'type' : [torch.Tensor]}
		}	

		# Check if dataset exists, download or generate if necessary.
		self.source_dataset()

		if not params['initialization_only']:

			# Load all the .jsons, but image generation is done in __getitem__
			self.dataset=[]
			
	
			self.logger.info("Loading dataset as json into memory.")
			# Val and Test are not shuffled
			if self.set == 'val' or self.set == 'test':
				for tasklist in os.listdir(self.data_folder_child):
					if tasklist[4:-8] in self.tasks:
						with gzip.open(os.path.join(self.data_folder_child,tasklist)) as f:
							fulltask = f.read().decode('utf-8').split('\n')
							for datapoint in fulltask:
								self.dataset.append(json.loads(datapoint))
						print("{} task examples loaded.".format(tasklist[4:-8]))
					else:
						self.logger.info("Skipped loading {} task.".format(tasklist[4:-8]))
		
			# Training set is shuffled
			elif self.set == 'train':
				for zipfile in os.listdir(self.data_folder_child):
					with gzip.open(os.path.join(self.data_folder_child,zipfile)) as f:
						fullzip = f.read().decode('utf-8').split('\n')
						for datapoint in fullzip:
							task = json.loads(datapoint)
							if task['family'] in self.tasks:
								self.dataset.append(task)
					print("Zip file {} loaded.".format(zipfile))		

			self.length = len(self.dataset)

			# Testing output classes
			#if self.set == 'val':
			#	self.output_words = []
			#	for datapoint in self.dataset:
			#		for answer in datapoint['answers']:
			#			if not answer in self.output_words:
			#				self.output_words.append(answer)

				#print(self.output_words)
				#print(len(self.output_words) )

		else:
			self.logger.info("COG initialization complete.")
			exit(0)

		self.categories = ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor',
									 'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist',
									 'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape',
									 'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 'ExistShape',
									 'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'GetColor', 'GetColorSpace',
									 'GetShape', 'GetShapeSpace', 'SimpleCompareColor', 'SimpleCompareShape', 'AndSimpleExistColorGo', 'AndSimpleExistGo', 'AndSimpleExistShapeGo', 'CompareColorGo',
								 'CompareShapeGo', 'ExistColorGo', 'ExistColorSpaceGo', 'ExistGo', 'ExistShapeGo',
								 'ExistShapeSpaceGo', 'ExistSpaceGo', 'Go', 'GoColor', 'GoColorOf', 'GoShape',
								 'GoShapeOf', 'SimpleCompareColorGo', 'SimpleCompareShapeGo', 'SimpleExistColorGo',
								 'SimpleExistGo','SimpleExistShapeGo']


		self.tuple_list = [[0,0,0] for _ in range(len(self.categories))]
		self.categories_stats = dict(zip(self.categories, self.tuple_list))

	def evaluate_loss(self, data_dict, logits):
		"""
		Calculates accuracy equal to mean number of correct predictions in a given batch.
		The function calculates two separate losses for answering and pointing actions and sums them up.


		:param data_dict: DataDict({'targets_pointing', 'targets_answer', ...}).

		:param logits: Predictions being output of the model, consisting of a tuple (logits_answer, logits_pointing).
		"""
		# Get targets.
		targets_answer = data_dict['targets_answer']
		targets_pointing = data_dict['targets_pointing']

		# Get predictions.
		preds_answer = logits[0]
		preds_pointing = logits[1]

		# Get sizes.
		batch_size = logits[0].size(0)
		img_seq_len = logits[0].size(1)

		# Retrieve "pointing" masks, both of size [BATCH_SIZE x IMG_SEQ_LEN] and transform it into floats.
		mask_pointing = data_dict['masks_pnt'].type(self.app_state.FloatTensor)

		# Classification loss.
		# Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
		preds_answer = preds_answer.view(batch_size*img_seq_len, -1)
		# Reshape targets [BATCH_SIZE * IMG_SEQ_LEN]
		targets_answer = targets_answer.view(batch_size*img_seq_len)
		# Calculate loss.
		# Ignore_index: specifies a target VALUE that is ignored and does not contribute to the input gradient. 
		# -1 is set when we do not use that action.
		ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
		self.loss_answer = ce_loss_fn(preds_answer, targets_answer)

		# Pointing loss.
		# We will softmax over the third dimension of [BATCH_SIZE x IMG_SEQ_LEN x NUM_POINT_ACTIONS].
		logsoftmax_fn = nn.LogSoftmax(dim=2)
		# Calculate cross entropy [BATCH_SIZE x IMG_SEQ_LEN].
		ce_point = torch.sum((-targets_pointing * logsoftmax_fn(preds_pointing)), dim=2) * mask_pointing
		#print("mask_pointing =", mask_pointing)
		#print("ce_point = ", ce_point)

		# Calculate mean - manually, skipping all non-pointing elements of the targets.
		if mask_pointing.sum().item() != 0:
			self.loss_pointing = torch.sum(ce_point) / mask_pointing.sum() 
		else:
			self.loss_pointing = torch.tensor(0).type(self.app_state.FloatTensor)

		# Both losses are averaged over batch size and sequence lengts - so we can simply sum them.
		return self.loss_answer + self.loss_pointing


	def calculate_accuracy(self, data_dict, logits):
		""" Calculates accuracy equal to mean number of correct predictions in a given batch.
		WARNING: Applies mask to both logits and targets!

		:param data_dict: DataDict({'sequences', 'sequences_length', 'targets', 'mask'}).

		:param logits: Predictions being output of the model.

		"""
		# Get targets.
		targets_answer = data_dict['targets_answer']
		targets_pointing = data_dict['targets_pointing']

		# Get predictions.
		preds_answer = logits[0]
		preds_pointing = logits[1]

		# Get sizes.
		batch_size = logits[0].size(0)
		img_seq_len = logits[0].size(1)

		# Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
		preds_answer = preds_answer.view(batch_size*img_seq_len, -1)
		preds_pointing = preds_pointing.view(batch_size*img_seq_len, -1)

		# Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
		targets_answer = targets_answer.view(batch_size*img_seq_len)
		# Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
		targets_pointing = targets_pointing.view(batch_size*img_seq_len, -1)

		# Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
		mask_answer = data_dict['masks_word']
		mask_answer = mask_answer.view(batch_size*img_seq_len)

		mask_pointing = data_dict['masks_pnt']
		mask_pointing = mask_pointing.view(batch_size*img_seq_len)

		#print("targets_answer = ", targets_answer)
		#print("preds_answer = ", preds_answer)
		#print("mask_answer = ", mask_answer)

		#print("targets_pointing = ", targets_pointing)
		#print("preds_pointing = ", preds_pointing)
		#print("mask_pointing = ", mask_pointing)


		#########################################################################
		# Calculate accuracy for Answering task.
		# Get answers [BATCH_SIZE * IMG_SEQ_LEN]
		_, indices = torch.max(preds_answer, 1)

		# Calculate correct answers with additional "masking".
		correct_answers = (indices == targets_answer).type(self.app_state.ByteTensor) * mask_answer

		# Calculate accurary.
		if mask_answer.sum() > 0:
			acc_answer = float(correct_answers.sum().item()) / float(mask_answer.sum().item())
		else:
			acc_answer = 0.0

		#########################################################################
		# Calculate accuracy for Pointing task.

        # Normalize pointing with softmax.
		softmax_pointing = nn.Softmax(dim=1)
		preds_pointing=softmax_pointing(preds_pointing)

		# Calculate mean square error for every pointing action.
		diff_pointing=(targets_pointing-preds_pointing)
		diff_pointing=diff_pointing**2
		# Sum all differences for a given answer.
		# As a results we got 1D tensor of size [BATCH_SIZE * IMG_SEQ_LEN].
		diff_pointing=torch.sum(diff_pointing,dim=1)

        # Apply  threshold.
		threshold=0.15**2
		
		# Check correct pointings.
		correct_pointing = (diff_pointing < threshold).type(self.app_state.ByteTensor) * mask_pointing
		#print('corect poitning',correct_pointing)
		# Calculate accurary.
		if mask_pointing.sum() > 0:
			acc_pointing = float(correct_pointing.sum().item()) / float(mask_pointing.sum().item())
		else:
			acc_pointing = 0.0


		#########################################################################
		# Total accuracy.
		acc_total = float(correct_answers.sum() + correct_pointing.sum()) / float( mask_answer.sum() + mask_pointing.sum() )
		#acc_total = torch.mean(torch.cat( (correct_answers.type(torch.FloatTensor), correct_pointing.type(torch.FloatTensor)) ) )

		# Return all three of them.
		return acc_total, acc_answer, acc_pointing



	def get_acc_per_family(self, data_dict, logits, categories_stats):
		"""
		Compute the accuracy per family for the current batch. Also accumulates
		the number of correct predictions & questions per family in self.correct_pred_families (saved
		to file).


		.. note::

			To refactor.


		:param data_dict: DataDict({'images','questions', 'questions_length', 'questions_string', 'questions_type', \
		'targets', 'targets_string', 'index','imgfiles'})
		:type data_dict: :py:class:`miprometheus.utils.DataDict`

		:param logits: network predictions.
		:type logits: :py:class:`torch.Tensor`

		"""

		# Get targets.
		targets_answer = data_dict['targets_answer']
		targets_pointing = data_dict['targets_pointing']

		#build dictionary to store acc families stats
		training=False
		if training:
			categories = ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor',
						  'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist',
						  'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape',
						  'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 'ExistShape',
						  'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'GetColor', 'GetColorSpace',
						  'GetShape', 'GetShapeSpace', 'SimpleCompareColor', 'SimpleCompareShape',
						  'AndSimpleExistColorGo', 'AndSimpleExistGo', 'AndSimpleExistShapeGo', 'CompareColorGo',
						  'CompareShapeGo', 'ExistColorGo', 'ExistColorSpaceGo', 'ExistGo', 'ExistShapeGo',
						  'ExistShapeSpaceGo', 'ExistSpaceGo', 'Go', 'GoColor', 'GoColorOf', 'GoShape',
						  'GoShapeOf', 'SimpleCompareColorGo', 'SimpleCompareShapeGo', 'SimpleExistColorGo',
						  'SimpleExistGo', 'SimpleExistShapeGo']

			tuple_list = [[0, 0, 0] for _ in range(len(self.categories))]
			categories_stats = dict(zip(categories, tuple_list))


		#Get tasks
		tasks = data_dict['tasks']

		# Get predictions.
		preds_answer = logits[0]
		preds_pointing = logits[1]

		# Get sizes.
		batch_size = logits[0].size(0)
		img_seq_len = logits[0].size(1)

		# Reshape predictions [BATCH_SIZE * IMG_SEQ_LEN x CLASSES]
		preds_answer = preds_answer.view(batch_size * img_seq_len, -1)
		preds_pointing = preds_pointing.view(batch_size * img_seq_len, -1)

		# Reshape targets: answers [BATCH_SIZE * IMG_SEQ_LEN]
		targets_answer = targets_answer.view(batch_size * img_seq_len)
		# Reshape targets: pointings [BATCH_SIZE * IMG_SEQ_LEN x NUM_ACTIONS]
		targets_pointing = targets_pointing.view(batch_size * img_seq_len, -1)

		# Retrieve "answer" and "pointing" masks, both of size [BATCH_SIZE * IMG_SEQ_LEN].
		mask_answer = data_dict['masks_word']
		mask_answer_non_flatten = mask_answer
		mask_answer = mask_answer.view(batch_size * img_seq_len)

		mask_pointing = data_dict['masks_pnt']
		mask_pointing_non_flatten = mask_pointing
		mask_pointing = mask_pointing.view(batch_size * img_seq_len)

		#########################################################################
		# Calculate accuracy for Answering task.
		# Get answers [BATCH_SIZE * IMG_SEQ_LEN]
		_, indices = torch.max(preds_answer, 1)

		# Calculate correct answers with additional "masking".
		correct_answers = (indices == targets_answer).type(self.app_state.ByteTensor) * mask_answer

		#########################################################################
		# Calculate accuracy for Pointing task.

		# Normalize pointing with softmax.
		softmax_pointing = nn.Softmax(dim=1)
		preds_pointing = softmax_pointing(preds_pointing)

		# Calculate mean square error for every pointing action.
		diff_pointing = (targets_pointing - preds_pointing)
		diff_pointing = diff_pointing ** 2
		# Sum all differences for a given answer.
		# As a results we got 1D tensor of size [BATCH_SIZE * IMG_SEQ_LEN].
		diff_pointing = torch.sum(diff_pointing, dim=1)

		# Apply  threshold.
		threshold = 0.15 ** 2

		# Check correct pointings.
		correct_pointing = (diff_pointing < threshold).type(self.app_state.ByteTensor) * mask_pointing

        #count correct and total for each category
		for i in range(batch_size):

			# update # of questions for the corresponding family

			#classification
			correct_ans = correct_answers.view(batch_size, img_seq_len, -1)
			categories_stats[tasks[i]][1] += float(correct_ans[i].sum().item())

			#pointing
			correct_pointing_non_flatten = correct_pointing.view(batch_size, img_seq_len, -1)
			categories_stats[tasks[i]][1] += float(correct_pointing_non_flatten[i].sum().item())

			#update the # of correct predictions for the corresponding family

			# classification
			categories_stats[tasks[i]][0] += float(mask_answer_non_flatten[i].sum().item())

			# pointing
			categories_stats[tasks[i]][0] += float(mask_pointing_non_flatten[i].sum().item())

			#put task accuracy in third position of the dictionary
			if categories_stats[tasks[i]][0]==0:
				categories_stats[tasks[i]][2] = 0.0

			else:
				categories_stats[tasks[i]][2] = categories_stats[tasks[i]][1]/categories_stats[tasks[i]][0]
        
		return categories_stats



	def output_class_to_int(self,targets_answer):
		#for j, target in enumerate(targets_answer):
		targets_answer = [-1 if a == 'invalid' else self.output_vocab.index(a) for a in targets_answer]
		targets_answer = torch.LongTensor(targets_answer)
		return targets_answer


	def __getitem__(self, index):
		"""
		Getter method to access the dataset and return a sample.

		:param index: index of the sample to return.
		:type index: int

		:return: ``DataDict({'images', 'questions', 'targets', 'targets_label'})``, with:
		
			- ``images``: Sequence of images,
			- ``tasks``: Which task family sample belongs to,
			- ``questions``: Question on the sequence (this is constant per sequence for COG),
			- ``targets_pointing``: Sequence of targets as tuple of floats for pointing tasks,
			- ``targets_answer``: Sequence of word targets for classification tasks.

		"""
		# This returns:
		# All variables are numpy array of float32
			# in_imgs: (n_epoch*batch_size, img_size, img_size, 3)
			# in_rule: (max_seq_length, batch_size) the rule language input, type int32
			# seq_length: (batch_size,) the length of each task instruction
			# out_pnt: (n_epoch*batch_size, n_out_pnt)
			# out_pnt_xy: (n_epoch*batch_size, -2)
			# out_word: (n_epoch*batch_size, n_out_word)
			# mask_pnt: (n_epoch*batch_size)
			# mask_word: (n_epoch*batch_size)		

		# Get values from JSON.
		(in_imgs, _, _, out_pnt, _, _, mask_pnt, mask_word, _) = jti.json_to_feeds([self.dataset[index]])
				
		# Create data dictionary.
		data_dict = self.create_data_dict()

		# Images [BATCH_SIZE x IMG_SEQ_LEN x DEPTH x HEIGHT x WIDTH].
		images = ((torch.from_numpy(in_imgs)).permute(1,0,4,2,3)).squeeze()
		data_dict['images']	= images

		# Set masks used in loss/accuracy calculations.
		data_dict['masks_pnt']	= torch.from_numpy(mask_pnt).type(torch.ByteTensor)
		data_dict['masks_word']	= torch.from_numpy(mask_word).type(torch.ByteTensor)

		data_dict['tasks']	= self.dataset[index]['family']
		data_dict['questions'] = [self.dataset[index]['question']]

		data_dict['questions_string'] = [self.dataset[index]['question']]
		data_dict['questions'] = torch.LongTensor([self.input_vocab.index(word) for word in data_dict['questions'][0].split()])
		if(data_dict['questions'].size(0) <= self.nwords):
			prev_size = data_dict['questions'].size(0)
			data_dict['questions'].resize_(self.nwords)
			data_dict['questions'][prev_size:] = 0

		# Set targets - depending on the answers.
		answers = self.dataset[index]['answers']
		data_dict['answers_string'] = self.dataset[index]['answers']
		if data_dict['tasks'] in self.classification_tasks:
			data_dict['targets_answer'] = self.output_class_to_int(answers)
		else :
			data_dict['targets_answer'] = torch.LongTensor([-1 for target in answers])
	
		# Why are we always setting pointing targets, and answer targets only when required (-1 opposite)?
		data_dict['targets_pointing'] = torch.FloatTensor(out_pnt)

		return data_dict

	def collate_fn(self, batch):
		"""
		Combines a list of :py:class:`miprometheus.utils.DataDict` (retrieved with :py:func:`__getitem__`) into a batch.

		:param batch: individual :py:class:`miprometheus.utils.DataDict` samples to combine.
		:type batch: list

		:return: ``DataDict({'images', 'tasks', 'questions', 'targets_pointing', 'targets_answer'})`` containing the batch.

		"""
		data_dict = self.create_data_dict()
		
		data_dict['images'] = torch.stack([sample['images'] for sample in batch]).type(self.app_state.FloatTensor)
		data_dict['tasks']  = [sample['tasks'] for sample in batch]
		data_dict['questions'] = torch.stack([sample['questions'] for sample in batch]).type(self.app_state.LongTensor)
		# Targets.
		data_dict['targets_pointing'] = torch.stack([sample['targets_pointing'] for sample in batch]).type(self.app_state.FloatTensor)
		data_dict['targets_answer'] = torch.stack([sample['targets_answer'] for sample in batch]).type(self.app_state.LongTensor)
		# Masks.
		data_dict['masks_pnt']	= torch.stack([sample['masks_pnt'] for sample in batch]).type(self.app_state.ByteTensor)
		data_dict['masks_word']	= torch.stack([sample['masks_word'] for sample in batch]).type(self.app_state.ByteTensor)
		data_dict['vocab'] = self.output_vocab

        #strings question and answer
		data_dict['questions_string'] = [question['questions_string'] for question in batch]
		data_dict['answers_string'] = [answer['answers_string'] for answer in batch]

		return data_dict

	def parse_tasks_and_dataset_type(self, params):
		"""
		Parses the task list and dataset type. Then sets folder paths to appropriate values.

		:param params: Dictionary of parameters (read from the configuration ``.yaml`` file).
		:type params: :py:class:`miprometheus.utils.ParamInterface`

		"""

		self.classification_tasks = ['AndCompareColor', 'AndCompareShape', 'AndSimpleCompareColor',
									 'AndSimpleCompareShape', 'CompareColor', 'CompareShape', 'Exist',
									 'ExistColor', 'ExistColorOf', 'ExistColorSpace', 'ExistLastColorSameShape',
									 'ExistLastObjectSameObject', 'ExistLastShapeSameColor', 'ExistShape',
									 'ExistShapeOf', 'ExistShapeSpace', 'ExistSpace', 'GetColor', 'GetColorSpace',
									 'GetShape', 'GetShapeSpace', 'SimpleCompareColor', 'SimpleCompareShape']

		self.regression_tasks = ['AndSimpleExistColorGo', 'AndSimpleExistGo', 'AndSimpleExistShapeGo', 'CompareColorGo',
								 'CompareShapeGo', 'ExistColorGo', 'ExistColorSpaceGo', 'ExistGo', 'ExistShapeGo',
								 'ExistShapeSpaceGo', 'ExistSpaceGo', 'Go', 'GoColor', 'GoColorOf', 'GoShape',
								 'GoShapeOf', 'SimpleCompareColorGo', 'SimpleCompareShapeGo', 'SimpleExistColorGo',
								 'SimpleExistGo','SimpleExistShapeGo']

		self.binary_tasks = ['AndCompareColor','AndCompareShape','AndSimpleCompareColor','AndSimpleCompareShape','CompareColor','CompareShape','Exist',
'ExistColor','ExistColorOf','ExistColorSpace','ExistLastColorSameShape','ExistLastObjectSameObject','ExistLastShapeSameColor',
'ExistShape','ExistShapeOf','ExistShapeSpace','ExistSpace','SimpleCompareColor','SimpleCompareShape'] 

		self.all_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'] + [
    'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige',
    'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white']

		self.all_shapes = ['circle', 'square', 'cross', 'triangle', 'vbar', 'hbar'] + list(string.ascii_lowercase)
		
		self.all_spaces = ['left', 'right', 'top', 'bottom']

		self.all_whens = ['now','latest','last1']

		self.input_vocab = ['invalid','.', ',', '?','object', 'color', 'shape','loc', 'on','if','then', 'else','exist','equal', 'and','the', 'of', 'with','point'] + self.all_spaces + self.all_colors + self.all_shapes + self.all_whens
		self.output_vocab = ['true','false'] + self.all_colors + self.all_shapes

		self.tasks = params['tasks']
		if self.tasks == 'class':
			self.tasks = self.classification_tasks
		elif self.tasks == 'reg':
			self.tasks = self.regression_tasks
			self.output_vocab = []
		elif self.tasks == 'all':
			self.tasks = self.classification_tasks + self.regression_tasks
		elif self.tasks == 'binary':
			self.tasks = self.binary_tasks
			self.output_vocab = ['true','false']

		self.input_words = len(self.input_vocab)
		self.output_classes = len(self.output_vocab)

		# If loading a default dataset, set default path names and set sequence length		
		if self.dataset_type == 'canonical':
			self.examples_per_task = 227280
			self.sequence_length = 4
			self.memory_length = 3
			self.max_distractors = 1
		elif self.dataset_type == 'hard':
			self.examples_per_task = 227280
			self.sequence_length = 8
			self.memory_length = 7
			self.max_distractors = 10
		elif self.dataset_type == 'generated':
			self.params.add_default_params({'generation':{'nr_processors':1}})
			try:
				self.examples_per_task = int(params['generation']['examples_per_task'])
				self.sequence_length = int(params['generation']['sequence_length'])
				self.memory_length = int(params['generation']['memory_length'])
				self.max_distractors = int(params['generation']['max_distractors'])
				self.nr_processors = int(params['generation']['nr_processors'])
			except KeyError:
				self.logger.info("Please specify examples per task, sequence length, memory length and maximum distractors "
					  "for a generated dataset under 'dataset_type'.")
				exit(1)
			except ValueError:
				self.logger.info("Examples per task, sequence length, memory length, maximum distractors and nr_processors "
					  "(if provided) must be of type int.")
				exit(2)

		self.dataset_name = str(self.sequence_length)+'_'+str(self.memory_length)+'_'+str(self.max_distractors)
		self.data_folder_parent = os.path.join(self.data_folder_main,'data_'+self.dataset_name) 
		self.data_folder_child = os.path.join(self.data_folder_parent,self.set+'_'+self.dataset_name)
		
	def source_dataset(self):
		"""
		Handles downloading and unzipping the canonical or hard version of the dataset.

		"""
		self.download = False
		if self.dataset_type == 'generated':
			self.download = self.check_and_download(self.data_folder_child)
			if self.download:			
				from miprometheus.problems.seq_to_seq.vqa.cog.cog_utils import generate_dataset
				generate_dataset.main(self.data_folder_parent,
															self.examples_per_task, 
															self.sequence_length, 
															self.memory_length, 
															self.max_distractors,
															self.nr_processors)
				self.logger.info('\nDataset generation complete for {}!'.format(self.dataset_name))
				self.download = False

		if self.dataset_type == 'canonical':
			self.download = self.check_and_download(self.data_folder_child, 
												  'https://storage.googleapis.com/cog-datasets/data_4_3_1.tar')
		
		elif self.dataset_type == 'hard':
			self.download = self.check_and_download(self.data_folder_child,
												  'https://storage.googleapis.com/cog-datasets/data_8_7_10.tar')
		if self.download:
			self.logger.info('\nDownload complete. Extracting...')
			tar = tarfile.open(os.path.expanduser('~/data/downloaded'))
			tar.extractall(path=self.data_folder_main)
			tar.close()
			self.logger.info('\nDone! Cleaning up.')
			os.remove(os.path.expanduser('~/data/downloaded'))
			self.logger.info('\nClean-up complete! Dataset ready.')

	def add_statistics(self, stat_col):
		"""
        Add :py:class:`COG`-specific stats to :py:class:`miprometheus.utils.StatisticsCollector`.

        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.

        """
		stat_col.add_statistic('loss_answer', '{:12.10f}')
		stat_col.add_statistic('loss_pointing', '{:12.10f}')
		stat_col.add_statistic('acc', '{:12.10f}')
		stat_col.add_statistic('acc_answer', '{:12.10f}')
		stat_col.add_statistic('acc_pointing', '{:12.10f}')
		stat_col.add_statistic('AndCompareColor', '{:12.10f}')
		stat_col.add_statistic('AndCompareShape', '{:12.10f}')
		stat_col.add_statistic('AndSimpleCompareColor', '{:12.10f}')
		stat_col.add_statistic('AndSimpleCompareShape', '{:12.10f}')
		stat_col.add_statistic('CompareColor', '{:12.10f}')
		stat_col.add_statistic('CompareShape', '{:12.10f}')
		stat_col.add_statistic('Exist', '{:12.10f}')
		stat_col.add_statistic('ExistColor', '{:12.10f}')
		stat_col.add_statistic('ExistColorOf', '{:12.10f}')
		stat_col.add_statistic('ExistColorSpace', '{:12.10f}')
		stat_col.add_statistic('ExistLastColorSameShape', '{:12.10f}')
		stat_col.add_statistic('ExistLastObjectSameObject', '{:12.10f}')
		stat_col.add_statistic('ExistLastShapeSameColor', '{:12.10f}')
		stat_col.add_statistic('ExistShape', '{:12.10f}')
		stat_col.add_statistic('ExistShapeOf', '{:12.10f}')
		stat_col.add_statistic('ExistShapeSpace', '{:12.10f}')
		stat_col.add_statistic('ExistSpace', '{:12.10f}')
		stat_col.add_statistic('GetColor', '{:12.10f}')
		stat_col.add_statistic('GetColorSpace', '{:12.10f}')
		stat_col.add_statistic('GetShape', '{:12.10f}')
		stat_col.add_statistic('GetShapeSpace', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareShape', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareColor', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareShape', '{:12.10f}')
		stat_col.add_statistic('AndSimpleExistColorGo', '{:12.10f}')
		stat_col.add_statistic('AndSimpleExistGo', '{:12.10f}')
		stat_col.add_statistic('AndSimpleExistShapeGo', '{:12.10f}')
		stat_col.add_statistic('CompareColorGo', '{:12.10f}')
		stat_col.add_statistic('CompareShapeGo', '{:12.10f}')
		stat_col.add_statistic('ExistColorGo', '{:12.10f}')
		stat_col.add_statistic('ExistColorSpaceGo', '{:12.10f}')
		stat_col.add_statistic('ExistGo', '{:12.10f}')
		stat_col.add_statistic('ExistShapeGo', '{:12.10f}')
		stat_col.add_statistic('ExistShapeSpaceGo', '{:12.10f}')
		stat_col.add_statistic('ExistSpaceGo', '{:12.10f}')
		stat_col.add_statistic('Go', '{:12.10f}')
		stat_col.add_statistic('GoColor', '{:12.10f}')
		stat_col.add_statistic('GoColorOf', '{:12.10f}')
		stat_col.add_statistic('GoShape', '{:12.10f}')
		stat_col.add_statistic('GoShapeOf', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareColorGo', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareShapeGo', '{:12.10f}')
		stat_col.add_statistic('SimpleExistColorGo', '{:12.10f}')
		stat_col.add_statistic('SimpleExistGo', '{:12.10f}')
		stat_col.add_statistic('SimpleCompareShape', '{:12.10f}')
		stat_col.add_statistic('SimpleExistShapeGo', '{:12.10f}')

	def collect_statistics(self, stat_col, data_dict, logits):
		"""
        Collects dataset details.
        :param stat_col: :py:class:`miprometheus.utils.StatisticsCollector`.
        :param data_dict: :py:class:`miprometheus.utils.DataDict` containing targets.
        :param logits: Prediction of the model (:py:class:`torch.Tensor`)
        """
		# Additional loss.
		stat_col['loss_answer'] = self.loss_answer.cpu().item()
		stat_col['loss_pointing'] = self.loss_pointing.cpu().item()

		# Accuracies.
		acc_total, acc_answer, acc_pointing = self.calculate_accuracy(data_dict, logits)
		stat_col['acc'] = acc_total
		stat_col['acc_answer'] = acc_answer
		stat_col['acc_pointing'] = acc_pointing

		categories_stats = self.categories_stats


		categories_dic = self.get_acc_per_family(data_dict, logits, categories_stats)
		for key in categories_dic:
			stat_col[key] = categories_dic[key][2]

#		print(categories_dic)




if __name__ == "__main__":
	
	""" 
	Unit test that checks data dimensions match expected values, and generates an image.
	Checks one regression and one classification task.
	"""

	# Test parameters
	batch_size = 44
	sequence_nr = 1

	# Timing test parameters
	timing_test = False
	testbatches = 100

	# -------------------------

	# Define useful params
	from miprometheus.utils.param_interface import ParamInterface
	params = ParamInterface()
	tasks = ['AndCompareColor']


	params.add_config_params({'data_folder': os.path.expanduser('~/data/cog'),
							  'set': 'val',
							  'dataset_type': 'canonical',
							  'tasks': tasks})

	# Create problem - task Go
	cog_dataset = COG(params)

	# Get a sample - Go
	#sample = cog_dataset[0]

	# Test whether data structures match expected definitions
#	assert sample['images'].shape == torch.ones((4, 3, 112, 112)).shape
#	assert sample['tasks'] == ['Go']
	#assert sample['questions'] == ['point now beige u']
#	assert sample['targets_pointing'].shape == torch.ones((4,2)).shape
#	assert len(sample['targets_answer']) == 4
#	assert sample['targets_answer'][0] == ' '  

	# Get another sample - CompareColor
	sample2 = cog_dataset[1000]
	#print(repr(sample2))

	# Test whether data structures match expected definitions
#	assert sample2['images'].shape == torch.ones((4, 3, 112, 112)).shape
#	assert sample2['tasks'] == ['CompareColor']
	#assert sample2['questions'] == ['color of latest g equal color of last1 v ?']
#	assert sample2['targets_pointing'].shape == torch.ones((4,2)).shape
#	assert len(sample2['targets_answer']) == 4
#	assert sample2['targets_answer'][0] == 'invalid'  
	
	# Set up Dataloader iterator
	from torch.utils.data import DataLoader
	
	dataloader = DataLoader(dataset=cog_dataset, collate_fn=cog_dataset.collate_fn,
							batch_size=batch_size, shuffle=False, num_workers=8)

	# Display single sample (0) from batch.
	batch = next(iter(dataloader))

	# VQA expects 'targets', so change 'targets_answer' to 'targets'
	# Implement a data_dict.pop later.
	batch['targets'] = batch['targets_pointing']
	batch['targets_label'] = batch['targets_answer']

	# Convert image to uint8
	batch['images'] = batch['images']/(np.iinfo(np.uint16).max)*255

	# Show sample - Task 1
	cog_dataset.show_sample(batch,0,sequence_nr)

	# Show sample - Task 2
	cog_dataset.show_sample(batch,1,sequence_nr)	


	if timing_test:
		# Test speed of generating images vs preloading generated images.
		import time

		# Define params to load entire dataset - all tasks included
		params = ParamInterface()
		params.add_config_params({
			'data_folder': '~/data/cog/',
			'set': 'val',
			'dataset_type': 'canonical',
			'tasks': 'all'})

		preload = time.time()
		full_cog_canonical = COG(params)
		postload = time.time() 

		dataloader = DataLoader(dataset=full_cog_canonical, collate_fn=full_cog_canonical.collate_fn,
								batch_size=batch_size, shuffle=True, num_workers=8)

		prebatch = time.time()
		for i, batch in enumerate(dataloader):
			if i == testbatches:
				break
			if i% 100 == 0:
				print('Batch # {} - {}'.format(i, type(batch)))
		postbatch = time.time()
	
		print('Number of workers: {}'.format(dataloader.num_workers))
		print('Time taken to load the dataset: {}s'.format(postload - preload))	
		print('Time taken to exhaust {} batches for a batch size of {} with image generation: {}s'.format(testbatches,
																										  batch_size,
																										  postbatch-prebatch))
	
		# Test pregeneration and loading
		for i, batch in enumerate(dataloader):
			if i == testbatches:
				print('Finished saving {} batches'.format(testbatches))
				break
			if not os.path.exists(os.path.expanduser('~/data/cogtest')):
				os.makedirs(os.path.expanduser('~/data/cogtest'))
			np.save(os.path.expanduser('~/data/cogtest/'+str(i)),batch['images'])

		preload = time.time()
		for i in range(testbatches):
			mockload = np.fromfile(os.path.expanduser('~/data/cogtest/'+str(i)+'.npy'))
		postload = time.time()
		print('Generation time for {} batches: {}, Load time for {} batches: {}'.format(testbatches, postbatch-prebatch, 
												testbatches, postload-preload))

		print('Timing test completed, removing files.')
		for i in range(testbatches):
			os.remove(os.path.expanduser('~/data/cogtest/'+str(i)+'.npy'))

	print('Done!')


