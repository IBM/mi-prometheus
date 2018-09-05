#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""statistics_collector.py: contains class used for collection and export of statistics during training, validation and testing """
__author__ = "Tomasz Kornuta"

from collections import Mapping


class StatisticsCollector(Mapping):
    """
    Specialized class used for collection and export of statistics during
    training, validation and testing.

    Inherits `collections.Mapping`, thererefor offers functionality
    close to a `dict`.

    """

    def __init__(self):
        """
        Initialization - creates dictionaries for statistics and formatting, adds standard statistics (episode and loss).
        """
        super(StatisticsCollector, self).__init__()
        self.statistics = dict()
        self.formatting = dict()

        # Add default statistics with formatting.
        self.add_statistic('episode', '{:06d}')
        self.add_statistic('loss', '{:12.10f}')

    def add_statistic(self, key, formatting):
        """
        Add statistic to collector.

        :param key: Key of the statistic.
        :param formatting: Formatting that will be used when logging and exporting to CSV.

        """
        self.formatting[key] = formatting
        self.statistics[key] = -1

    def __getitem__(self, key):
        """
        Get statistics value for given key.

        :param key: Key to value in parameters.
        :return: Statistics value associated with given key.

        """
        return self.statistics[key]

    def __setitem__(self, key, value):
        """
        Add/overwrites value of statistic associated with a given key.

        :param key: Key to value in parameters.
        :param value: Statistics value associated with given key.

        """
        self.statistics[key] = value

    # def __delitem__(self, key):

    def __len__(self):
        """
        Returns "length" of statistics (i.e. number of tracked values).
        """
        return len(self.statistics.__len__)

    def __iter__(self):
        """
        Iterator.
        """
        return iter(self.statistics.__iter__)

    def initialize_csv_file(self, log_dir, filename):
        """
        Method creates new csv file and initializes it with a header produced
        on the base of statistics names.

        :param log_dir: Path to file.
        :param filename: Filename to be created.
        :return: File stream opened for writing.

        """
        # Iterate through keys and concatenate them.
        header_str = ''
        for key, value in self.statistics.items():
            header_str += key + ","
        # Remove last coma and add \n.
        header_str = header_str[:-1] + '\n'

        # Open file for writing.
        csv_file = open(log_dir + filename, 'w', 1)
        csv_file.write(header_str)

        return csv_file

    def export_statistics_to_csv(self, csv_file):
        """
        Method writes current statistics to csv using the possessed formatting.

        :param file: File stream opened for writing.

        """
        # Iterate through values and concatenate them.
        values_str = ''
        for key, value in self.statistics.items():
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')
            # Add value to string using formatting.
            values_str += format_str.format(value) + ","
        # Remove last coma and add \n.
        values_str = values_str[:-1] + '\n'
        csv_file.write(values_str)

    def export_statistics_to_string(self, additional_tag=''):
        """
        Method returns current statistics in the form of string using the
        possessed formatting.

        :return: String being concatenation of statistics names and values.

        """
        # Iterate through keys and values and concatenate them.
        stat_str = ''
        for key, value in self.statistics.items():
            stat_str += key + ' '
            # Get formatting - using '{}' as default.
            format_str = self.formatting.get(key, '{}')
            # Add value to string using formatting.
            stat_str += format_str.format(value) + "; "
        # Remove last two element.
        stat_str = stat_str[:-2] + " " + additional_tag
        return stat_str

# format_str = 'episode {:05d}; acc={:12.10f}; loss={:12.10f}; length={:d}'
# logger.info(format_str.format(episode, accuracy, loss, train_length))

    def export_statistics_to_tensorboard(self, tb_writer):
        """
        Method exports current statistics to tensorboard.

        :param tb_writer: TensorBoard writer.

        """
        # Get episode number.
        episode = self.statistics['episode']
        # Iterate through keys and values and concatenate them.
        stat_str = ''
        for key, value in self.statistics.items():
            # Skip episode.
            if key == 'episode':
                continue
            tb_writer.add_scalar(key, value, episode)

# training_writer.add_scalar('Loss', loss, episode)


if __name__ == "__main__":

    stat_col = StatisticsCollector()
    stat_col.add_statistic('acc', '{:2.3f}')

    stat_col['episode'] = 0
    stat_col['loss'] = 0.7
    stat_col['acc'] = 100

    csv_file = stat_col.initialize_csv_file('./', 'collector_test.csv')
    stat_col.export_statistics_to_csv(csv_file)
    print(stat_col.export_statistics_to_string())

    stat_col['episode'] = 1
    stat_col['loss'] = 0.7
    stat_col['acc'] = 99.3
    stat_col['seq_length'] = 5

    stat_col.export_statistics_to_csv(csv_file)
    print(stat_col.export_statistics_to_string('[Validation]'))
