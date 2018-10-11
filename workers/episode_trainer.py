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
episode_trainer.py:

    - This file contains the implementation of the ``EpisodeTrainer``, which inherits from ``Trainer``.

"""
__author__ = "Vincent Marois, Tomasz Kornuta"

from torch.nn.utils import clip_grad_value_

import argparse
import workers.worker as worker
import workers.trainer as trainer
from workers.trainer import Trainer


class EpisodeTrainer(Trainer):
    """
    Implementation for the episode-based Trainer.

    ..note::

        The default ``EpochTrainer`` is based on epochs. While an epoch can be defined for all finite-size datasets,\
         it makes less sense for problems which have a very large, almost infinite, dataset (like algorithmic \
         tasks, which generate random data on-the-fly). This is why this Episode Trainer is implemented.
         Instead of looping on epochs, it iterates directly on episodes (we call an iteration on a single batch\
          an episode).


    """

    def __init__(self, name="EpisodeTrainer"):
        """
        Only calls the ``Trainer`` constructor as the initialization phase is identical to the ``Trainer``.

       :param name: Name of the worker (DEFAULT: ''EpisodeTrainer'').

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(EpisodeTrainer, self).__init__(name)



    def setup_experiment(self):
        """
        Sets up experiment for episode trainer:

            - Calls base class setup_experiment to parse the command line arguments 
        """
        # Call base method to parse all command line arguments, load configuration, create problems and model etc.
        super(EpisodeTrainer, self).setup_experiment()


    def run_experiment(self):
        """
        Main function of the ``EpisodeTrainer``, runs the experiment.

        Iterates over the (cycled) DataLoader (one iteration = one episode).

        .. note::

            The test for terminal conditions (e.g. convergence) is done at the end of each episode. \
            The terminal conditions are as follows:

                 - The loss is below the specified threshold (using the validation loss or the highest training loss\
                  over several episodes),
                  - The maximum number of episodes has been met,
                  - The user pressed 'Stop experiment' during visualization (TODO: should change that)


        The function does the following for each episode:

            - Handles curriculum learning if set,
            - Resets the gradients
            - Forwards pass of the model,
            - Logs statistics and exports to TensorBoard (if set),
            - Computes gradients and update weights
            - Activate visualization if set,
            - Validate the model on a batch according to the validation frequency.
            - Checks the above terminal conditions.


        :param flags: Parsed arguments from the parser.

        """
        # Ask for confirmation - optional.
        if self.flags.confirm:
            input('Press any key to run the experiment')

        # Initialize tensorboard and statistics collection.
        self.initialize_statistics_collection()
        self.initialize_tensorboard()

        # cycle the DataLoader -> infinite iterator
        self.training_dataloader = self.cycle(self.training_dataloader)

        try:
            '''
            Main training and validation loop.
            '''
            episode = 0
            for training_dict in self.training_dataloader:

                # reset all gradients
                self.optimizer.zero_grad()

                # Check the visualization flag - Set it if visualization is wanted during training & validation episodes.
                if (0 <= self.flags.visualize <= 1):
                    self.app_state.visualize = True
                else:
                    self.app_state.visualize = False

                # Turn on training mode for the model.
                self.model.train()

                # 1. Perform forward step, get predictions and compute loss.
                logits, loss = self.predict_evaluate_collect(self.model, self.training_problem, 
                    training_dict, self.training_stat_col, episode)

                # 2. Backward gradient flow.
                loss.backward()

                # Check the presence of the 'gradient_clipping'  parameter.
                try:
                    # if present - clip gradients to a range (-gradient_clipping, gradient_clipping)
                    val = self.params['training']['gradient_clipping']
                    clip_grad_value_(self.model.parameters(), val)

                except KeyError:
                    # Else - do nothing.
                    pass

                # 3. Perform optimization.
                self.optimizer.step()

                # 4. Log collected statistics.

                # 4.1. Export to csv - at every step.
                self.training_stat_col.export_statistics_to_csv()

                # 4.2. Export data to tensorboard - at logging frequency.
                if (self.training_batch_writer is not None) and (episode % self.flags.logging_interval == 0):
                    self.training_stat_col.export_statistics_to_tensorboard()

                    # Export histograms.
                    if self.flags.tensorboard >= 1:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name, param.data.cpu().numpy(), episode, bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: data :: {}".format(name, e))

                    # Export gradients.
                    if self.flags.tensorboard >= 2:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), episode,
                                                                bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: grad :: {}".format(name, e))

                # 4.3. Log to logger - at logging frequency.
                if episode % self.flags.logging_interval == 0:
                    self.logger.info(self.training_stat_col.export_statistics_to_string())

                    # empty Statistics Collector to avoid memory leak
                    self.training_stat_col.empty()

                # 5. Check visualization of training data.
                if self.app_state.visualize:

                    # Allow for preprocessing
                    training_dict, logits = self.training_problem.plot_preprocessing(training_dict, logits)

                    # Show plot, if user will press Stop then a SystemExit exception will be thrown.
                    self.model.plot(training_dict, logits)


                #  6. Validate and (optionally) save the model.
                if (episode % self.model_validation_interval) == 0:

                    # Check visualization flag
                    if (1 <= self.flags.visualize <= 2):
                        self.app_state.visualize = True
                    else:
                        self.app_state.visualize = False

                    # Perform validation.
                    validation_loss = self.validate_on_batch(self.validation_batch, episode)

                    # Save the model using the latest validation statistics.
                    self.model.save(self.model_dir, self.validation_stat_col)

                # 7. Terminal conditions.

                # Apply curriculum learning - change some of the Problem parameters
                self.curric_done = self.training_problem.curriculum_learning_update_params(episode)

                # 7.2. - the loss is < threshold (only when curriculum learning is finished if set.)
                if self.curric_done or not self.must_finish_curriculum:

                    # loss_stop = True if convergence
                    loss_stop = validation_loss < self.params['training']['terminal_condition']['loss_stop']
                    # We already saved that model.

                    if loss_stop:
                        # Finish the training.
                        break

                # 7.3. - The episodes number limit has been reached.
                if episode+1 > self.params['training']['terminal_condition']['max_episodes']:
                    # If we reach this condition, then it is possible that the model didn't converge correctly
                    # and present poorer performance.

                    # We still save the model as it may perform better during this episode
                    # (as opposed to the previous episode)

                    # Validate on the problem if required - so we can collect the
                    # statistics needed during saving of the best model.
                    # Do not visualize.
                    self.app_state.visualize = False
                    self.validate_on_batch(self.validation_batch, episode)

                    # save the model
                    self.model.save(self.model_dir, self.validation_stat_col)

                    # "Finish" the training.
                    break

                # check if we are at the end of the 'epoch': Indicate that the DataLoader is now cycling.
                if ((episode + 1) % self.training_problem.get_epoch_size(
                        self.params['training']['problem']['batch_size'])) == 0:
                    self.logger.warning('The DataLoader has exhausted -> using cycle(iterable).')

                # Move on to next episode.
                episode += 1

            '''
            End of main training and validation loop.
            '''

            # Check visualization flag - turn on visualization for last validation if needed.
            if (self.flags.visualize == 3):
                self.app_state.visualize = True
            else:
                self.app_state.visualize = False

            # Validate over the entire validation set.
            self.validate_on_set(episode)

            # Save the model using the average validation loss.
            self.model.save(self.model_dir, self.validation_stat_agg)

            self.logger.info('Training finished!')

        except SystemExit:
            # the training did not end properly
            self.logger.warning('Training interrupted!')
        finally:
            # Finalize statistics collection.
            self.finalize_statistics_collection()
            self.finalize_tensorboard()


if __name__ == '__main__':

    episode_trainer = EpisodeTrainer()
    # parse args, load configuration and create all required objects.
    episode_trainer.setup_experiment()
    # GO!
    episode_trainer.run_experiment()