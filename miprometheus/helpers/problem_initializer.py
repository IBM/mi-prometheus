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
	- ProblemInitializer checks whether a dataset exists at a location, and downloads or generates if it doesn't.

"""

__author__ = "Emre Sevgen"

import os
import argparse
import urllib.request
import sys
import time
#import json

from miprometheus.problems.problem_factory import ProblemFactory
from miprometheus.workers import Worker

class ProblemInitializer(Worker):

	def __init__(self, config=None, name=None, path=None):
	"""
	Initialize ProblemInitializer, which runs the __init__ for a provided problem, optionally overriding some parameters.

	:param config: Path to a config file to initialize from.
	:type config: string
	:param name: Name of a problem to initialize using default parameters
	:type name:	string
	:param path: Path to initialize problem, overrides default data_folder if provided.
	:type path: string

	"""


		# Call base constructor to set up app state, registry and add default params.
		super(ProblemInitializer, self).__init__(name='ProblemInitializer', add_default_parser_args=False)

		# If no config is provided, try to build a problem from command line name.
		if config == None:
			self.params.add_default_params({'problem': {'name': name}})

		# If config is provided, parse and build problem from it.
		else:
			try:
				self.params.add_config_params_from_yaml(os.path.expanduser(config))
				self.params = self.params['training']
			except FileNotFoundError:
				print("Config file at path '{}' not found.".format(config))
				exit(1)

		# If path is provided, override default path.
		if path != None:
			self.params.add_config_params({'problem': {'data_folder': path}})

		# Build Problem
		try:
			_ = ProblemFactory.build(self.params['problem'])
		except AttributeError:
			print("Provided problem name not found.")
			exit(1)
			
		

# Function to make check and download easier
def CheckAndDownload(filefoldertocheck, url='none', downloadname='downloaded'):
	"""
	Checks whether a file or folder exists at given path (relative to storage folder), otherwise downloads files from given URL.

	:param filefoldertocheck: Relative path to a file or folder to check to see if it exists.
	:type filefoldertocheck: string
	:param url: URL to download files from.
	:type url: string
	:param downloadname: What to name the downloaded file. (DEFAULT: "downloaded").
	:type downloadname: string.

	:return: False if file was found, True if a download was necessary.

	"""
	if not ( os.path.isfile  ( os.path.join(self.path,filefoldertocheck)) or 
					 os.path.isdir( os.path.join(self.path,filefoldertocheck)) ):
		print('Downloading {}'.format(url))
		urllib.request.urlretrieve(url, os.path.join(self.path,downloadname), reporthook)
		return True
	else:
		print('Dataset found at {}'.format(os.path.join(self.path,filefoldertocheck)))
		return False

# Progress bar function
def reporthook(count, block_size, total_size):
	global start_time
	if count == 0:
		  start_time = time.time()
		  return
	duration = time.time() - start_time
	progress_size = int(count * block_size)
	speed = int(progress_size / (1024 * duration))
	percent = int(count * block_size * 100 / total_size)
	sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
		              (percent, progress_size / (1024 * 1024), speed, duration))
	sys.stdout.flush()

# Useful function to properly parse argparse
def int_or_str(val):
	try:
		return int(val)
	except:
		return val

# Useful function to properly parse argparse
def str_to_bool(val):
	if val.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif val.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
	""" May be used as a script to prepare a dataset for subsequent runs. """

	# Create parser with a list of runtime arguments.
	parser = argparse.ArgumentParser(description=
	'Initializes any problem, thereby downloading any prerequisite datasets if required. \n' + 
	'A dataset can be initialize either from a config file with --c, or from the command line directly ' +
	'with --problem to initialize a problem with its default parameters. An optional --path argument' +
	'can be provided to override default problem path.'
	, formatter_class=argparse.RawTextHelpFormatter)

	parser.add_argument('--c', type=str,
	help='A config file to initialize from.')

	parser.add_argument('--problem', type=str,
	help='Initialize a problem with its default parameters directly from the command line')

	parser.add_argument('--path', type=str,
	help='Change from problem default path to this path when initializing')

  # Add a command line dict parser
	#parser.add_argument('--options', type=json.loads,
	#help='A dictionary to initialize from, obtained directly from the command line.' + 
	#'If --c is provided, arguments here will override the config file parameters.')
	
	args = parser.parse_args()

	if args.c == None and args.problem == None:
		print("Please provide either a config file or a problem name.")
		exit(1)

	elif args.c != None and args.problem != None:
		print("Both a config and a problem name is provided. Please only provide one or the other.")
		exit(1)	

	ProblemInitializer(args.c, args.problem, args.path)
