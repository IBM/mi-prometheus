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
statistics_estimators.py: Allows to compute several statistical estimators (e.g. average, standard deviation...)\
 using the statistics collected over an epoch or a validation phase by the ``StatisticsCollector``.

 Allows to summarize the current epoch or validation phase using statistical estimators.

 """
__author__ = "Vincent Marois"

from utils.statistics_collector import StatisticsCollector


class StatisticsEstimators(StatisticsCollector):
    """
    Specialized class used for the computation of several statistical estimators.

    Inherits from ``StatisticsCollector`` as it extends its capabilities: it relies \
    on ``StatisticsCollector`` to collect the statistics over an epoch (training set) \
    or a validation (over the validation set).

    Once the statistics have been collected, the ``StatisticsEstimator`` allows to compute several \
    statistical estimators to summarize the last epoch or validation phase.

    E.g. With the list of loss values from the last epoch, we can compute the average loss, the min & max, \
    and the standard deviation.


    """

    def __init__(self):
        """
        Constructor for the ``StatisticsEstimator``.

        Add the following basic statistical estimators:

            - Minimum & maximum loss value,
            - Average loss,
            - Standard deviation of the loss.

        Other statistical estimators can be added via ``self.add_estimator()``.

        """
        # call base constructor
        super(StatisticsEstimators, self).__init__()

        self.estimators = dict()

        # add 'estimators' for the episode index and epoch
        self.add_estimator('episode', '{:06d}')
        self.add_estimator('epoch', '{:06d}')

        # Add default statistical estimators for the loss (indicating a formatting).
        self.add_estimator('loss', '{:12.10f}')  # represents the average loss
        self.add_estimator('loss_min', '{:12.10f}')
        self.add_estimator('loss_max', '{:12.10f}')
        self.add_estimator('loss_std', '{:12.10f}')

    def add_estimator(self, key, formatting):
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
        self.estimators[key] = -1

    def __getitem__(self, key):
        """
        Get the values list of the specified statistical estimator.

        :param key: Name of the statistical estimator to get the values list of.
        :type key: str

        :return: Values list associated with the specified statistical estimator.

        """
        return self.estimators[key]

    def __setitem__(self, key, value):
        """
        Set the value of the specified statistical estimator, thus overwriting the existing one.

        :param key: Name of the statistical estimator to set the value of.
        :type key: str

        :param value: Value to set for the given key.
        :type value: int, float

        """
        self.estimators[key] = value

    def __delitem__(self, key):
        """
        Delete the specified statistical estimator.

        :param key: Key to be deleted.
        :type key: str

        """
        del self.estimators[key]

    def __len__(self):
        """
        Returns the number of tracked statistical estimators.
        """
        return self.estimators.__len__()

    def __iter__(self):
        """
        Return an iterator on the currently tracked statistical estimators.
        """
        return self.estimators.__iter__()

    def initialize_csv_file(self, log_dir, filename):
        """
        This method creates a new `csv` file and initializes it with a header produced \
        on the base of the statistical estimators names.

        :param log_dir: Path to file.
        :type log_dir: str

        :param filename: Filename to be created.
        :type filename: str

        :return: File stream opened for writing.

        """
        header_str = ''

        # Iterate through keys and concatenate them.
        for key in self.estimators.keys():
            header_str += key + ","

        # Remove last coma and add \n.
        header_str = header_str[:-1] + '\n'

        # Open file for writing.
        csv_file = open(log_dir + filename, 'w', 1)
        csv_file.write(header_str)

        return csv_file

    def export_estimators_to_csv(self, csv_file):
        """
        This method writes the current statistical estimators values to the `csv_file` using the associated formatting.

        :param csv_file: File stream opened for writing.

        """
        values_str = ''

        # Iterate through values and concatenate them.
        for key, value in self.estimators.items():

            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')

            # Add value to string using formatting.
            values_str += format_str.format(value) + ","

        # Remove last coma and add \n.
        values_str = values_str[:-1] + '\n'

        csv_file.write(values_str)

    def export_estimators_to_string(self, additional_tag=''):
        """
        This method returns the current statistical estimators values in the form of a string using the \
        associated formatting.

        :param additional_tag: An additional tag to append at the end of the created string.
        :type additional_tag: str


        :return: String being the concatenation of the statistical estimators names & values.

        """
        stat_str = ''

        # Iterate through keys and values and concatenate them.
        for key, value in self.estimators.items():

            stat_str += key + ' '
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')
            # Add value to string using formatting.
            stat_str += format_str.format(value) + "; "

        # Remove last two element and add additional tag
        stat_str = stat_str[:-2] + " " + additional_tag

        return stat_str

    def export_estimators_to_tensorboard(self, tb_writer):
        """
        Method exports current statistical estimators values to TensorBoard.

        :param tb_writer: TensorBoard writer.
        :type tb_writer: ``tensorboardX.SummaryWriter``

        """
        # Get episode number.
        episode = self.estimators['episode']

        # Iterate through keys and values and concatenate them.
        for key, value in self.estimators.items():
            # Skip episode.
            if key == 'episode':
                continue
            tb_writer.add_scalar(key, value, episode)


if __name__ == "__main__":

    stat_est = StatisticsEstimators()

    # create some random values
    import random
    import numpy as np
    loss_values = random.sample(range(100), 100)

    # add episode value and delete epoch statistics
    stat_est.statistics['episode'].append(0)

    del stat_est.statistics['epoch']

    stat_est['loss_min'] = min(loss_values)
    stat_est['loss_max'] = max(loss_values)
    stat_est['loss'] = np.average(loss_values)
    stat_est['loss_std'] = np.std(loss_values)

    print(stat_est.export_estimators_to_string())

    stat_est.add_estimator('acc_mean', '{:2.5f}')
    stat_est['acc_mean'] = np.mean(random.sample(range(100), 100))

    print(stat_est.export_estimators_to_string('[Epoch 1]'))
