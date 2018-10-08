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
statistics_aggregator.py: Allows to compute several statistical metrics (e.g. average, standard deviation...)\
 using the statistics collected over an epoch or a validation phase by the ``StatisticsCollector``.

 """
__author__ = "Vincent Marois"

from utils.statistics_collector import StatisticsCollector


class StatisticsAggregator(StatisticsCollector):
    """
    Specialized class used for the computation of several statistics measures.

    Inherits from ``StatisticsCollector`` as it extends its capabilities: it relies \
    on ``StatisticsCollector`` to collect the statistics over an epoch or a validation \
    phase (over the validation set). Once the statistics have been collected, the \
    ``StatisticsAggregator`` allows to compute several statistics measures to summarize the \
    last epoch or validation phase. E.g. With the list of loss values from the last epoch, \
    we can compute the average loss, the min & max, and the standard deviation.


    """

    def __init__(self):
        """
        Constructor for the ``StatisticsAggregator``.

        TODO: complete doc.
        """
        # call base constructor
        super(StatisticsAggregator, self).__init__()

        self.aggregated_statistics = dict()

        # Add default loss statistics (and formatting).
        self.add_aggregated_statistics('loss_mean', '{:12.10f}')
        self.add_aggregated_statistics('loss_min', '{:12.10f}')
        self.add_aggregated_statistics('loss_max', '{:12.10f}')
        self.add_aggregated_statistics('loss_std', '{:12.10f}')

    def add_aggregated_statistics(self, key, formatting):
        """
        Add a statistics metric to the aggregator.
        The value associated to the specified key is initiated as an empty list.

        :param key: Statistical metric to add. Such statistical metric (e.g. min, max, mean, std...)\
         should be based on an existing statistics collected by the ``StatisticsCollector`` \
         (e.g. added by ``add_statistic`` and collected by ``model.collect_statistics`` or \
         ``problem.collect_statistics``.
        :type key: str

        :param formatting: Formatting that will be used when logging and exporting to CSV.
        :type formatting: str

        """
        self.formatting[key] = formatting

        # instantiate associated value as list.
        self.aggregated_statistics[key] = list()

    def __getitem__(self, key):
        """
        Get aggregated statistics value for the given key.

        :param key: Name of the aggregated statistics to get the value of.
        :type key: str

        :return: Aggregated statistics value list associated with the given key.

        """
        return self.aggregated_statistics[key]

    def __setitem__(self, key, value):
        """
        Add value to the list of the aggregated statistic associated with a given key.

        :param key: Name of the aggregated statistics to add the value to.
        :type key: str

        :param value: Aggregated statistics value to append to the list associated with the given key.
        :type value: int, float

        """
        self.aggregated_statistics[key].append(value)

    def __delitem__(self, key):
        """
        Delete the specified aggregated key.

        :param key: Key to be deleted.
        :type key: str

        """
        del self.aggregated_statistics[key]

    def __len__(self):
        """
        Returns "length" of ``self.aggregated_statistics`` (i.e. number of tracked values).
        """
        return self.aggregated_statistics.__len__()

    def __iter__(self):
        """
        Iterator on the aggregated statistics.
        """
        return self.aggregated_statistics.__iter__()

    def aggregate_statistics(self, stat_col, model, problem):
        """
        Aggregates statistics, i.e. compute specified statistics for the metrics collected by the ``stat_col``.\

        :param stat_col: ``StatisticsCollector`` that collects some metrics such as the loss, episode etc.

        :param model: Model class which possesses a ``collect_aggregated_statistics`` (and also \
        ``add_statistics``, ``collect_statistics``.)
        :type model: ``models.model.Model`` or a subclass.

        :param problem: Problem class which possesses a ``collect_aggregated_statistics`` (and also \
        ``add_statistics``, ``collect_statistics``.)
        :type problem: ``problems.problem.Problem`` or a subclass.

        """
        model.collect_aggregated_statistics(stat_col)
        problem.collect_aggregated_statistics(stat_col)

    def initialize_csv_file(self, log_dir, filename):
        """
        This method creates a new csv file and initializes it with a header produced \
        on the base of the aggregated statistics names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.

        """
        header_str = ''

        # Iterate through keys and concatenate them.
        for key in self.aggregated_statistics.keys():
            header_str += key + ","

        # Remove last coma and add \n.
        header_str = header_str[:-1] + '\n'

        # Open file for writing.
        csv_file = open(log_dir + filename, 'w', 1)
        csv_file.write(header_str)

        return csv_file

    def export_statistics_to_csv(self, csv_file):
        """
        This method writes the current aggregated statistics to the `csv_file` using the associated formatting.

        :param csv_file: File stream opened for writing.

        """
        # Iterate through values and concatenate them.
        values_str = ''
        for key, value in self.aggregated_statistics.items():

            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Add value to string using formatting.
            values_str += format_str.format(value[-1]) + ","

        # Remove last coma and add \n.
        values_str = values_str[:-1] + '\n'

        csv_file.write(values_str)

    def export_statistics_to_string(self, additional_tag=''):
        """
        This method returns the current aggregated statistics in the form of string using the \
        associated formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str


        :return: String being the concatenation of statistics names and values.

        """
        # Iterate through keys and values and concatenate them.
        stat_str = ''
        for key, value in self.aggregated_statistics.items():

            stat_str += key + ' '
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')
            # Add value to string using formatting.
            stat_str += format_str.format(value[-1]) + "; "

        # Remove last two element and add additional tag
        stat_str = stat_str[:-2] + " " + additional_tag

        return stat_str


if __name__ == "__main__":

    stat_agg = StatisticsAggregator()

    # create some random values
    import random
    import numpy as np
    loss_values = random.sample(range(10), 100)

    stat_agg['loss_min'] = min(loss_values)
    stat_agg['loss_max'] = max(loss_values)
    stat_agg['loss_mean'] = np.average(loss_values)
    stat_agg['loss_std'] = np.std(loss_values)

    print(stat_agg.export_statistics_to_string())

    stat_agg.add_aggregated_statistics('acc_mean', '{:2.5f}')
    stat_agg['acc_mean'] = np.mean(random.sample(range(100), 100))

    print(stat_agg.export_statistics_to_string('[Epoch 1]'))
