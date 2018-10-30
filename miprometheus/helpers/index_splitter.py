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

"""
index_splitter.py:

    - Contains the definition of a new ``Helper`` class, called IndexSplitter.

"""
__author__ = "Tomasz Kornuta"

import os
from miprometheus.utils.split_indices import split_indices
from miprometheus.workers import Worker
from miprometheus.problems.problem_factory import ProblemFactory


class IndexSplitter(Worker):
    """
    Defines the ``IndexSplitter`` class.

    The class is responsible for generation of files with indices splitting given dataset into two.

    Those files can later be used in training/verification testing when using ``SubsetRandomSampler``.

    .. note::

        General usage:

            -- user provides the output dir where files with indices will be created (`--o`)

            -- user provides the problem name (`--p`) OR length of the dataset (`--l`)
        
            -- user provides split `--s` (value from 1 to l-2, which are border cases when one or the other \
            split will contain a single index)
        
        Additionally, the user might turn random sampling on or off  by --r (DEFAULT: ``True``)

            -- when random_sampling is on, both files will contain (exclusive) random lists of indices

            -- when off, both files will contain ranges, i.e. [0, s-1] and [s, l-1] respectively.

 
    """
    def __init__(self, name="IndexSplitter"):
        """
        Set parser arguments.

        ..note::
            As it does not really share any functionality with other basic workers, it does not call the base ``Worker`` constructor. 

       :param name: Name of the worker (DEFAULT: "IndexSplitter").
       :type name: str

        """
        # Call base constructor to set up app state, registry and add default params.
        super(IndexSplitter, self).__init__(name=name, add_default_parser_args=False)

        # Add arguments to the specific parser.
        # These arguments will be shared by all basic workers.
        self.parser.add_argument('--outdir',
                                 dest='outdir',
                                 type=str,
                                 default=".",
                                 help='Path to the output directory where the files with indices will be stored.'
                                      ' (DEFAULT: .)')

        self.parser.add_argument('--problem',
                                 dest='problem_name',
                                 type=str,
                                 default='',
                                 help='Name of the problem to be splitted. (WARNING: exclusive with --l)')

        self.parser.add_argument('--length',
                                 dest='length',
                                 type=int,
                                 default=-1,
                                 help='Length (size) of the dataset (WARNING: exclusive with --p)')

        self.parser.add_argument('--split',
                                 dest='split',
                                 type=int,
                                 default=-1,
                                 help='Value indicating number of indices/samples in the first set. '
                                      'Value from 1 to length-2 are accepted. The two border cases mean that one '
                                      'or the other split will contain a single index)')

        self.parser.add_argument('--random_sampling_off',
                                 dest='random_sampling_off',
                                 default=False,
                                 action='store_true',
                                 help='When on, both files will contain (exclusive) random lists of indices. '
                                      'When off, both files will contain ranges, i.e. [0, split-1] and '
                                      '[split, length-1] respectively')

    def run(self):
        """
        Creates two files with splits.

            - Parses command line arguments.

            - Loads the problem class (if required).

            - Generates two lists (or ranges) of exclusive indices.

            - Writes those lists to two separate files.

        """
        # Parse arguments.
        self.flags, self.unparsed = self.parser.parse_known_args()

        # Display results of parsing.
        self.display_parsing_results()

        # Get output dir.
        self.out_dit = self.flags.outdir
        # Create - just in case.
        os.makedirs(self.out_dit, exist_ok=True)

        # Check if we can estimate length.
        if self.flags.problem_name == '' and self.flags.length == -1:
            self.logger.error('Index splitter operates on length (size) of the problem, '
                              'please set problem (--p) or its length (--l).')
            exit(-1)

        # Check if user pointed only one of them.
        if self.flags.problem_name != '' and self.flags.length != -1:
            self.logger.error('Flags problem (--p) and length (--l) are exclusive, please use only one of them.')
            exit(-2)

        # Check if user set the split.
        if self.flags.split == -1:
            self.logger.error('Please set the split (--s).')
            exit(-3)
        split = self.flags.split

        # Build the problem.
        if self.flags.problem_name != '':
            self.params.add_default_params({'name': self.flags.problem_name})
            problem = ProblemFactory.build(self.params)
            length = len(problem)
        else:
            length = self.flags.length

        # Check the splitting.
        if split < 1 or split > length-1:
            self.logger.error("Split must lie within 1 to {}-2 range, which are border cases "
                              "when one or the other split will contain a single index.".format(length))
            exit(-4)

        self.logger.info("Splitting dataset of length {} into splits of size {} and {}.".format(length, split, length - split))

        # Split the indices.
        split_a, split_b = split_indices(length, split, self.logger, self.flags.random_sampling_off == False)

        # Write splits to files.
        name_a = os.path.expanduser(self.flags.outdir)+'split_a.txt'
        split_a.tofile(name_a, sep=",", format="%s")

        # Write splits to files.
        name_b = os.path.expanduser(self.flags.outdir)+'split_b.txt'
        split_b.tofile(name_b, sep=",", format="%s")

        self.logger.info("Splits written to {} ({} indices) and {} ({} indices).".format(name_a, len(split_a), name_b, len(split_b)))
        # Finished.


def main():
    """
    Entry point function for the ``IndexSplitter``.

    """
    worker = IndexSplitter()
    # parse args and do the splitting.
    worker.run()


if __name__ == '__main__':

    main()
