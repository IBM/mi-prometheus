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
estimator.py: Allows to compute several statistical aggregators (e.g. average, standard deviation...)\
 using the statistics collected over an epoch or a validation phase by the ``StatisticsCollector``.

 Allows to summarize the current epoch or validation phase using statistical aggregators.

 """
__author__ = "Vincent Marois"

from utils.statistics_collector import StatisticsCollector


class StatisticsAggregator(StatisticsCollector):
    """
    Specialized class used for the computation of several statistical aggregators.

    Inherits from ``StatisticsCollector`` as it extends its capabilities: it relies \
    on ``StatisticsCollector`` to collect the statistics over an epoch (training set) \
    or a validation (over the validation set).

    Once the statistics have been collected, the ``StatisticsEstimator`` allows to compute several \
    statistical aggregators to summarize the last epoch or validation phase.

    E.g. With the list of loss values from the last epoch, we can compute the average loss, the min & max, \
    and the standard deviation.


    """

    def __init__(self):
        """
        Constructor for the ``StatisticsEstimator``.

        Add the following basic statistical aggregators:

            - Minimum & maximum loss value,
            - Average loss,
            - Standard deviation of the loss.

        Other statistical aggregators can be added via ``self.add_aggregator()``.

        """
        # call base constructor
        super(StatisticsAggregator, self).__init__()

        self.aggregators = dict()

        # add 'aggregators' for the episode.
        self.add_aggregator('episode', '{:06d}')
        # Number of aggregated episodes.
        self.add_aggregator('episodes_aggregated', '{:06d}')

        # Add default statistical aggregators for the loss (indicating a formatting).
        # Represents the average loss, but stying with loss for TensorBoard "variable compatibility".
        self.add_aggregator('loss', '{:12.10f}')  
        self.add_aggregator('loss_min', '{:12.10f}')
        self.add_aggregator('loss_max', '{:12.10f}')
        self.add_aggregator('loss_std', '{:12.10f}')

    def add_aggregator(self, key, formatting):
        """
        Add a statistical estimator.
        The value associated to the specified key is initiated as -1.

        :param key: Statistical estimator to add. Such estimator (e.g. min, max, mean, std...)\
         should be based on an existing statistics collected by the ``StatisticsCollector`` \
         (e.g. added by ``add_statistic`` and collected by ``model.collect_statistics`` or \
         ``problem.collect_statistics``.
        :type key: str

        :param formatting: Formatting that will be used when logging and exporting to CSV.
        :type formatting: str

        """
        self.formatting[key] = formatting

        # instantiate associated value as list.
        self.aggregators[key] = -1

    def __getitem__(self, key):
        """
        Get the values list of the specified statistical estimator.

        :param key: Name of the statistical estimator to get the values list of.
        :type key: str

        :return: Values list associated with the specified statistical estimator.

        """
        return self.aggregators[key]

    def __setitem__(self, key, value):
        """
        Set the value of the specified statistical estimator, thus overwriting the existing one.

        :param key: Name of the statistical estimator to set the value of.
        :type key: str

        :param value: Value to set for the given key.
        :type value: int, float

        """
        self.aggregators[key] = value

    def __delitem__(self, key):
        """
        Delete the specified statistical estimator.

        :param key: Key to be deleted.
        :type key: str

        """
        del self.aggregators[key]

    def __len__(self):
        """
        Returns the number of tracked statistical aggregators.
        """
        return self.aggregators.__len__()

    def __iter__(self):
        """
        Return an iterator on the currently tracked statistical aggregators.
        """
        return self.aggregators.__iter__()

    def aggregate_statistics(self, stat_col):
        """
        Method aggregates the default statistics collected by the Statistics Collector.

        :param: stat_col: ''StatisticsCollector''
        """
        # By default, copy last values for all variables have mathing names.
        # (will work well for e.g. episode or epoch)
        for k,v in stat_col.items():
            if k in self.aggregators:
                # Copy last collected value.
                self.aggregators[k] = v[-1]
        # Simply copy the last episode from collector.
        #self.aggregators['episode'] = stat_col['episode'][-1]

        # Get loss values.
        loss_values = stat_col['loss']
        # Calcualte default aggregates.
        self.aggregators['loss'] = np.average(loss_values)
        self.aggregators['loss_min'] = min(loss_values)
        self.aggregators['loss_max'] = max(loss_values)
        self.aggregators['loss_std'] = np.std(loss_values)
        self.aggregators['episodes_aggregated'] = len(loss_values)
        

    def initialize_csv_file(self, log_dir, filename):
        """
        This method creates a new `csv` file and initializes it with a header produced \
        on the base of the statistical aggregators names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.

        """
        header_str = ''

        # Iterate through keys and concatenate them.
        for key in self.aggregators.keys():
            header_str += key + ","

        # Remove last coma and add \n.
        header_str = header_str[:-1] + '\n'

        # Open file for writing.
        csv_file = open(log_dir + filename, 'w', 1)
        csv_file.write(header_str)

        return csv_file

    def export_aggregators_to_csv(self, csv_file):
        """
        This method writes the current statistical aggregators values to the `csv_file` using the associated formatting.

        :param csv_file: File stream opened for writing.

        """
        values_str = ''

        # Iterate through values and concatenate them.
        for key, value in self.aggregators.items():

            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Add value to string using formatting.
            values_str += format_str.format(value) + ","

        # Remove last coma and add \n.
        values_str = values_str[:-1] + '\n'

        csv_file.write(values_str)

    def export_aggregators_to_string(self, additional_tag=''):
        """
        This method returns the current statistical aggregators values in the form of a string using the \
        associated formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str


        :return: String being the concatenation of the statistical aggregators names & values.

        """
        stat_str = ''

        # Iterate through keys and values and concatenate them.
        for key, value in self.aggregators.items():

            stat_str += key + ' '
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')
            # Add value to string using formatting.
            stat_str += format_str.format(value) + "; "

        # Remove last two elements and add additional tag
        stat_str = stat_str[:-2] + " " + additional_tag

        return stat_str

    def export_aggregators_to_tensorboard(self, tb_writer):
        """
        Method exports current statistical aggregators values to TensorBoard.

        :param tb_writer: TensorBoard writer.
        :type tb_writer: ``tensorboardX.SummaryWriter``

        """
        # Get episode number.
        episode = self.aggregators['episode']

        # Iterate through keys and values and concatenate them.
        for key, value in self.aggregators.items():
            # Skip episode.
            if key == 'episode':
                continue
            tb_writer.add_scalar(key, value, episode)


if __name__ == "__main__":

    stat_col = StatisticsCollector()
    stat_agg = StatisticsAggregator()

    import random
    import numpy as np

    # create some random values
    loss_values = random.sample(range(100), 100)
    # "Collect" basic statistics.
    for episode, loss in enumerate(loss_values):
        stat_col['episode'] = episode
        stat_col['loss'] = loss
        #print(stat_col.export_statistics_to_string())
        
    # Aggregate.
    stat_agg.aggregate_statistics(stat_col)
    print(stat_agg.export_aggregators_to_string())

    # Add new aggregator (a simulation of "additional statistics collected by model")
    stat_agg.add_aggregator('acc_mean', '{:2.5f}')
    stat_agg['acc_mean'] = np.mean(random.sample(range(100), 100))

    print(stat_agg.export_aggregators_to_string('[Epoch 1]'))
