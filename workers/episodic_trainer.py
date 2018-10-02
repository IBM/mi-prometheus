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
base_trainer.py:

    - This file sets hosts a function which adds specific arguments a trainer will need.
    - Also defines the BaseTrainer() class.


"""
__author__ = "Vincent Marois"

import argparse
import collections
from torch.nn.utils import clip_grad_value_

import workers.base_worker as worker
import workers.base_trainer as trainer
from workers.base_trainer import BaseTrainer
from utils.worker_utils import forward_step, validation, cycle


class EpisodicTrainer(BaseTrainer):
    """
    Implementation for the episodic Trainer.

    ..note::

        The default Trainer is based on epochs. While an epoch can be defined for all finite-size datasets,\
         it makes less sense for problems which have a very large, almost infinite, dataset (like algorithmic \
         tasks, which generate random data on-the-fly). This is why this episodic Trainer is implemented.
         Instead of looping on epochs, it iterates directly on episodes (we call an iteration on a single batch\
          an episode).


    """

    def __init__(self, flags: argparse.Namespace):
        """
        Only calls the ``BaseTrainer`` constructor as the initialization phase is identical to the ``BaseTrainer``.

        :param flags: Parsed arguments from the parser.

        """
        self.name = 'EpisodicTrainer'
        super(EpisodicTrainer, self).__init__(flags=flags)

        # delete 'epoch' entry in the StatisticsCollector
        self.stat_col.__delitem__('epoch')

    def forward(self, flags: argparse.Namespace):
        """
        TODO: Make documentation

        :param flags: Parsed arguments from the parser.

        """
        # Ask for confirmation - optional.
        if flags.confirm:
            input('Press any key to continue')

        # Start Training
        last_losses = collections.deque()

        # Flag denoting whether we converged (or reached last episode).
        terminal_condition = False

        # cycle the DataLoader -> infinite iterator
        self.problem = cycle(self.problem)

        '''
        Main training and validation loop.
        '''
        for episode, data_dict in enumerate(self.problem):

            # apply curriculum learning - change some of the Problem parameters
            self.curric_done = self.dataset.curriculum_learning_update_params(episode)

            # reset all gradients
            self.optimizer.zero_grad()

            # Check the visualization flag - Set it if visualization is wanted during training & validation episodes.
            if flags.visualize is not None and flags.visualize <= 1:
                self.app_state.visualize = True
            else:
                self.app_state.visualize = False

            # Turn on training mode for the model.
            self.model.train()

            # 1. Perform forward step, get predictions and compute loss.
            logits, loss = forward_step(self.model, self.dataset, episode, self.stat_col, data_dict)

            if not self.use_validation_problem:

                # Store the calculated loss on a list.
                last_losses.append(loss)

                # Truncate list length.
                if len(last_losses) > self.loss_length:
                    last_losses.popleft()

            # 2. Backward gradient flow.
            loss.backward()

            # Check the presence of the 'gradient_clipping'  parameter.
            try:
                # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
                val = self.param_interface['training']['gradient_clipping']
                clip_grad_value_(self.model.parameters(), val)

            except KeyError:
                # Else - do nothing.
                pass

            # 3. Perform optimization.
            self.optimizer.step()

            # 4. Log statistics.

            # Log to logger.
            self.logger.info(self.stat_col.export_statistics_to_string())

            # Export to csv.
            self.stat_col.export_statistics_to_csv(self.training_file)

            # Export data to tensorboard.
            if (flags.tensorboard is not None) and (episode % flags.logging_frequency == 0):
                self.stat_col.export_statistics_to_tensorboard(self.training_writer)

                # Export histograms.
                if flags.tensorboard >= 1:
                    for name, param in self.model.named_parameters():
                        try:
                            self.training_writer.add_histogram(name, param.data.cpu().numpy(), episode, bins='doane')

                        except Exception as e:
                            self.logger.error("  {} :: data :: {}".format(name, e))

                # Export gradients.
                if flags.tensorboard >= 2:
                    for name, param in self.model.named_parameters():
                        try:
                            self.training_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), episode,
                                                               bins='doane')

                        except Exception as e:
                            self.logger.error("  {} :: grad :: {}".format(name, e))

            # Check visualization of training data.
            if self.app_state.visualize:

                # Allow for preprocessing
                data_dict, logits = self.dataset.plot_preprocessing(data_dict, logits)

                # Show plot, if user presses Quit - break.
                if self.model.plot(data_dict, logits):
                    break

            #  5. Validate and (optionally) save the model.
            user_pressed_stop = False

            if (episode % self.model_validation_interval) == 0:

                # Validate on the problem if required.
                if self.use_validation_problem:

                    # Check visualization flag
                    if flags.visualize is not None and (flags.visualize == 1 or flags.visualize == 2):
                        self.app_state.visualize = True
                    else:
                        self.app_state.visualize = False

                    # Perform validation.
                    validation_loss, user_pressed_stop = validation(self.model, self.problem_validation, episode,
                                                                    self.stat_col, self.data_valid, flags, self.logger,
                                                                    self.validation_file, self.validation_writer)

                # Save the model using the latest (validation or training) statistics.
                self.model.save(self.model_dir, self.stat_col)

            # 6. Terminal conditions.

            # I. The User pressed stop during visualization.
            if user_pressed_stop:
                break

            # II. & III - the loss is < threshold - only when we finished curriculum (# TODO: ?).
            if self.curric_done or not self.must_finish_curriculum:

                # break if conditions applied: convergence or max episodes
                loss_stop = False
                if self.use_validation_problem:
                    loss_stop = validation_loss < self.param_interface['training']['terminal_condition']['loss_stop']
                    # We already saved that model.
                else:
                    loss_stop = max(last_losses) < self.param_interface['training']['terminal_condition']['loss_stop']
                    # We already saved that model.

                if loss_stop:
                    # Ok, we have converged.
                    terminal_condition = True
                    # "Finish" the training.
                    break

            # IV - The episodes number limit has been reached.
            if episode == self.param_interface['training']['terminal_condition']['max_episodes']:
                terminal_condition = True
                # If we reach this condition, then it is possible that the model didn't converge correctly
                # and present poorer performance.

                # We still save the model as it may perform better during this episode
                # (as opposed to the previous episode)

                # Validate on the problem if required - so we can collect the
                # statistics needed during saving of the best model.
                if self.use_validation_problem:
                    validation_loss, user_pressed_stop = validation(self.model, self.problem_validation, episode,
                                                                    self.stat_col, self.data_valid, flags, self.logger,
                                                                    self.validation_file, self.validation_writer)
                # save the model
                self.model.save(self.model_dir, self.stat_col)

                # "Finish" the training.
                break

            # check if we are at the end of the 'epoch': Indicate that the DataLoader is now cycling.
            if ((episode + 1) % self.dataset.get_epoch_size(
                    self.param_interface['training']['problem']['batch_size'])) == 0:
                self.logger.warning('The DataLoader has exhausted -> using cycle(iterable).')

            # Move on to next episode.
            episode += 1

        '''
        End of main training and validation loop.
        '''

        # Check whether we have finished training properly.
        if terminal_condition:

            self.logger.info('Learning finished!')
            # Check visualization flag - turn on visualization for last validation if needed.
            if flags.visualize is not None and (flags.visualize == 3):
                self.app_state.visualize = True

                # Perform validation.
                if self.use_validation_problem:
                    _, _ = validation(self.model, self.problem_validation, episode, self.stat_col, self.data_valid,
                                      flags, self.logger, self.validation_file, self.validation_writer)

            else:
                self.app_state.visualize = False

        else:  # the training did not end properly
            self.logger.warning('Learning interrupted!')

        # Close all files.
        self.training_file.close()
        self.validation_file.close()

        if flags.tensorboard is not None:
            # Close the TensorBoard writers.
            self.training_writer.close()
            self.validation_writer.close()


if __name__ == '__main__':
    # Create parser with list of  runtime arguments.
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # add default arguments
    worker.add_arguments(argp)

    # add trainers-specific arguments
    trainer.add_arguments(argp)

    # Parse arguments.
    FLAGS, unparsed = argp.parse_known_args()

    episodic_trainer = EpisodicTrainer(FLAGS)
    episodic_trainer.forward(FLAGS)
