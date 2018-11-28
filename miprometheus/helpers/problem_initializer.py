    #!/usr/bin/env python30
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

"""
problem_initializer.py:

	- Contains the definition of a new ``Helper`` class, called ProblemInitializer.
	- ProblemInitializer runs __init__ for a given problem, downloading and/or generating its datasets as necessary.
	- Additionally, contains helpful function(s) to aid in checking and downloading data for problems.

"""

__author__ = "Emre Sevgen"

import os
import argparse
# import json

from miprometheus.problems.problem_factory import ProblemFactory
from miprometheus.workers import Worker


class ProblemInitializer(Worker):

	def __init__(self, config=None, name=None, path=None):
		"""
		Initialize :py:class:`ProblemInitializer`, which runs the :py:func:`__init__` for a provided \
		``Problem``, downloading and/or generating its datasets as necessary, optionally overriding some parameters.

		:param config: Path to a config file to initialize from.
		:type config: str

		:param name: Name of a problem to initialize using default parameters
		:type name:	str

		:param path: Path to initialize problem, overrides default data_folder if provided.
		:type path: str

		"""

		# Call base constructor to set up app state, registry and add default params.
		super(ProblemInitializer, self).__init__(name='ProblemInitializer', add_default_parser_args=False)

		# If no config is provided, try to build a problem from command line name.
		if config is None:
			self.params.add_default_params({'problem': {'name': name}})

		# If config is provided, parse and build problem from it.
		else:
			try:
				# self.params.add_config_params_from_yaml(os.path.expanduser(config))
				configs_to_load = self.recurrent_config_parse(os.path.expanduser(config),[])
				self.recurrent_config_load(configs_to_load)
				self.params = self.params['training']
			except FileNotFoundError:
				self.logger.error("Config file at path '{}' not found.".format(config))
				exit(1)

		# If path is provided, override default path.
		if path is not None:
			self.params.add_config_params({'problem': {'data_folder': path}})

		# Pass initialization only flag.
		self.params.add_config_params({'problem': {'initialization_only': True}})

		# Build Problem
		try:
			_ = ProblemFactory.build(self.params['problem'])
		except AttributeError:
			self.logger.error("Provided problem name not found.")
			exit(1)

	# Useful function to properly parse argparse
	@staticmethod
	def int_or_str(val):
		"""
		Try to return ``int(val)`` else return ``val``.

		:param val: Value to evaluate.

		"""
		try:
			return int(val)
		except ValueError:
			return val

	# Useful function to properly parse argparse
	@staticmethod
	def str_to_bool(val):
		"""
		Return ``True`` if val.lower() in ('yes', 'true', 't', 'y', '1').

		Return ``False`` if val.lower() in ('no', 'false', 'f', 'n', '0')

		Else raise argparse.ArgumentTypeError.

		:param val: Value to evaluate.

		"""
		if val.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif val.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
	""" May be used as a script to prepare a dataset for subsequent runs. """

	# Create parser with a list of runtime arguments.
	parser = argparse.ArgumentParser(description='Initializes any problem, thereby downloading any prerequisite '
												 'datasets if required. \nA dataset can be initialized either from a '
												 'config file with --c, or from the command line directly with --problem'
												 ' to initialize a problem with its default parameters. An optional '
												 '--path argument can be provided to override default problem path.',
									 formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('--c', type=str,
	help='A config file to initialize from.')

	parser.add_argument('--problem', type=str,
	help='Initialize a problem with its default parameters directly from the command line')

	parser.add_argument('--path', type=str,
	help='Change from problem default path to this path when initializing')

	# Add a command line dict parser
	# parser.add_argument('--options', type=json.loads,
	# help='A dictionary to initialize from, obtained directly from the command line.' +
	# 'If --c is provided, arguments here will override the config file parameters.')
	
	args = parser.parse_args()

	if args.c is None and args.problem is None:
		print("Please provide either a config file or a problem name.")
		exit(1)

	elif args.c is not None and args.problem is not None:
		print("Both a config and a problem name is provided. Please only provide one or the other.")
		exit(1)	

	ProblemInitializer(args.c, args.problem, args.path)
