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
classic_trainer.py:

    - This file contains the implementation of the ``OfflineTrainer``, which inherits from ``Trainer``. \
    The ``OfflineTrainer`` is based on epochs.

"""
__author__ = "Vincent Marois, Tomasz Kornuta"

import numpy as np
from torch.nn.utils import clip_grad_value_

from miprometheus.workers.trainer import Trainer


class OfflineTrainer(Trainer):
    """
    Implementation for the epoch-based ``OfflineTrainer``.

    ..note::

        The default ``OfflineTrainer`` is based on epochs. \
        An epoch is defined as passing through all samples of a finite-size dataset.\
        The ``OfflineTrainer`` allows to loop over all samples from the training set many times i.e. in many epochs. \
        When an epochs finishes, it performs a similar step for the validation set and collects the statistics.


    """

    def __init__(self, name="OfflineTrainer"):
        """
        Only calls the ``Trainer`` constructor as the initialization phase is identical to the ``Trainer``.

       :param name: Name of the worker (DEFAULT: "OfflineTrainer").
       :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(OfflineTrainer, self).__init__(name)

    def setup_experiment(self):
        """
        Sets up an experiment for the ``OfflineTrainer``:

            - Calls base class setup_experiment to parse the command line arguments,
            - Sets up the terminal conditions (loss threshold, episodes (optional) & epochs limits).

        """
        # Call base method to parse all command line arguments, load configuration, create problems and model etc.
        super(OfflineTrainer, self).setup_experiment()

        ################# TERMINAL CONDITIONS ################# 
        self.logger.info('Terminal conditions:\n' + '='*80)

        # Terminal condition I: loss. 
        self.params['training']['terminal_conditions'].add_default_params({'loss_stop': 1e-5})
        self.loss_stop = self.params['training']['terminal_conditions']['loss_stop']
        self.logger.info("Setting Loss Stop threshold to {}".format(self.loss_stop))

        # In this trainer, Partial Validation is optional.
        self.params['validation'].add_default_params({'partial_validation_interval': -1})
        self.partial_validation_interval = self.params['validation']['partial_validation_interval']
        if self.partial_validation_interval <= 0:
            self.logger.info("Partial Validation deactivated")
        else:
            self.logger.info("Partial Validation activated with interval equal to {} episodes".format(self.partial_validation_interval))

        # Terminal condition II: max epochs. Mandatory.
        self.params["training"]["terminal_conditions"].add_default_params({'epoch_limit': 10})
        self.epoch_limit = self.params["training"]["terminal_conditions"]["epoch_limit"]
        if self.epoch_limit <= 0:
            self.logger.error("OffLine Trainer relies on epochs, thus Epoch Limit must be a positive number!")
            exit(-5)
        else:
            self.logger.info("Setting the Epoch Limit to: {}".format(self.epoch_limit))

        # Calculate the epoch size in terms of episodes.
        self.epoch_size = len(self.training_dataloader)
        self.logger.info('Epoch size in terms of training episodes: {}'.format(self.epoch_size))

        # Terminal condition III: max episodes. Optional.
        self.params["training"]["terminal_conditions"].add_default_params({'episode_limit': -1})
        self.episode_limit = self.params['training']['terminal_conditions']['episode_limit']
        if self.episode_limit < 0:
            self.logger.info("Termination based on Episode Limit is disabled")
            # Set to infinity.
            self.episode_limit = np.Inf
        else:
            self.logger.info("Setting the Episode Limit to: {}".format(self.episode_limit))
        self.logger.info('\n' + '='*80)

        # Export and log configuration, optionally asking the user for confirmation.
        self.export_experiment_configuration(self.log_dir, "training_configuration.yaml", self.flags.confirm)

    def run_experiment(self):
        """
        Main function of the ``Trainer``.

        Iterates over the number of epochs of the training set.

        .. note::

            Because of the export of stats, weights and gradients to TensorBoard, we need to\
            keep track of the current episode index from the start of the training, even \
            though the Worker runs on epoch.

        .. warning::

            The test for terminal conditions (e.g. convergence) is done at the end of each epoch. \
            The terminal conditions are as follows:

                - I. The loss is below the specified threshold (using the full validation loss),
                - TODO: II. Early stopping is set and the full validation loss did not change by delta \
                    for the indicated number of epochs,
                - III. The maximum number of epochs has been met,
                - IV. The maximum number of episodes has been met (optional).

            Besides, the user can always stop experiment by pressing 'Stop experiment' during visualization.


        The function does the following for each epoch:

            - Executes the ``initialize_epoch()`` & ``finish_epoch()`` function of the ``Problem`` class,
            - For each episode:

                    - Resets the gradients,
                    - Forwards pass of the model,
                    - Logs statistics and exports to TensorBoard (if set),
                    - Computes gradients and update weights,
                    - Activates visualization if set (vis. level 0),
                    - Validates the model on a batch according to the validation frequency.

            - At the end of epoch:

                    - Handles curriculum learning (if set),
                    - Validates the model on the full validation set, logs the statistics \
                    and visualizes on a random batch if set (vis. level 1 or 2)
                    - Checks the above terminal conditions.

        The last validation on the full set is done additionally at the end on training, \
        with optional visualization of a random batch if set (vis. level 3).

        """
        # Initialize TensorBoard and statistics collection.
        self.initialize_statistics_collection()
        self.initialize_tensorboard()

        try:
            '''
            Main training and validation loop.
            '''
            # Reset the counter.
            episode = 0
            last_epoch = 0

            # Set default termination cause.
            termination_cause = "Epoch limit reached"
            # Iterate over epochs.
            for epoch in range(self.epoch_limit):
                self.logger.info('Starting next epoch: {}'.format(epoch))
                # Inform the training problem class that epoch has started.
                self.training_problem.initialize_epoch(epoch)

                # Exhaust training set.
                for training_dict in self.training_dataloader:

                    # reset all gradients
                    self.optimizer.zero_grad()

                    # Check the visualization flag - Set it if visualization is wanted during
                    # training & validation episodes.
                    if 0 <= self.flags.visualize <= 1:
                        self.app_state.visualize = True
                    else:
                        self.app_state.visualize = False

                    # Turn on training mode for the model.
                    self.model.train()

                    # 1. Perform forward step, get predictions and compute loss.
                    logits, loss = self.predict_evaluate_collect(self.model, self.training_problem, 
                                                                 training_dict, self.training_stat_col, episode, epoch)

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
                    self.training_stat_col.export_to_csv()

                    # 4.2. Export data to tensorboard - at logging frequency.
                    if (self.training_batch_writer is not None) and (episode % self.flags.logging_interval == 0):
                        self.training_stat_col.export_to_tensorboard()

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
                        self.logger.info(self.training_stat_col.export_to_string())

                    # 5. Check visualization of training data.
                    if self.app_state.visualize:

                        # Allow for preprocessing
                        training_dict, logits = self.training_problem.plot_preprocessing(training_dict, logits)

                        # Show plot, if user will press Stop then a SystemExit exception will be thrown.
                        self.model.plot(training_dict, logits)

                    #  6. Validate and (optionally) save the model.
                    if (self.partial_validation_interval > 0) and (episode % self.partial_validation_interval) == 0:

                        # Check visualization flag
                        if 1 <= self.flags.visualize <= 2:
                            self.app_state.visualize = True
                        else:
                            self.app_state.visualize = False

                        # Perform validation.
                        self.validate_on_batch(self.validation_batch, episode, epoch)

                        # Save the model using the latest validation statistics.
                        self.model.save(self.model_dir, self.validation_stat_col)

                    # III. The episodes number limit has been reached.
                    if episode+1 >= self.episode_limit:
                        termination_cause = "Episode Limit reached"
                        last_epoch = epoch
                        break

                    # Move on to next episode.
                    episode += 1

                # Epoch just ended!
                # Inform the problem class that the epoch has ended.
                self.training_problem.finalize_epoch(epoch)

                # Aggregate training statistics for the epoch.
                self.aggregate_and_export_statistics(self.model, self.training_problem, 
                                                     self.training_stat_col, self.training_stat_agg,
                                                     episode, '[Epoch {}]'.format(epoch))

                # Apply curriculum learning - change some of the Problem parameters
                self.curric_done = self.training_problem.curriculum_learning_update_params(episode)

                # Perform full validation!

                # Check visualization flag - turn on visualization for last validation if needed.
                if 1 <= self.flags.visualize <= 2:
                    self.app_state.visualize = True
                else:
                    self.app_state.visualize = False

                # Validate over the entire validation set.
                self.validate_on_set(episode, epoch)

                # Save the model using the average validation loss.
                self.model.save(self.model_dir, self.validation_stat_agg)

                # Terminal conditions.
                # I - the loss is < threshold (only when curriculum learning is finished if set.)
                # We check that condition only in validation step!
                if self.curric_done or not self.must_finish_curriculum:

                    # Check the Full Validation loss.
                    if self.validation_stat_agg["loss"] < self.loss_stop:
                        termination_cause = "Full Validation Loss went below Loss Stop threshold (model converged)"
                        last_epoch = epoch
                        break

                # II. Early stopping is set and loss hasn't improved by delta in n epochs.
                # early_stopping(index=epoch, avg_valid_loss). (TODO: coming in next release)
                # termination_cause = 'Early Stopping.'

                # IV. The epoch number limit has been reached, condition is already made in for loop.
                last_epoch = epoch

            '''
            End of main training and validation loop. Perform final full validation.
            '''
            self.logger.info('\n' + '='*80)
            self.logger.info('Training finished because {}'.format(termination_cause))
            # Check visualization flag - turn on visualization for last validation if needed.
            if self.flags.visualize == 3:
                self.app_state.visualize = True
            else:
                self.app_state.visualize = False

            # Validate over the entire validation set.
            self.validate_on_set(episode, last_epoch)

            # Save the model using the average validation loss.
            self.model.save(self.model_dir, self.validation_stat_agg)

            self.logger.info('Experiment finished!')

        except SystemExit as e:
            # the training did not end properly
            self.logger.error('Experiment interrupted because {}'.format(e))
        except KeyboardInterrupt:
            # the training did not end properly
            self.logger.error('Experiment interrupted!')
        finally:
            # Finalize statistics collection.
            self.finalize_statistics_collection()
            self.finalize_tensorboard()


def main():
    """
    Entry point function for the ``OfflineTrainer``.

    """
    trainer = OfflineTrainer()
    # parse args, load configuration and create all required objects.
    trainer.setup_experiment()
    # GO!
    trainer.run_experiment()


if __name__ == '__main__':

    main()
