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
online_trainer.py:

    - This file contains the implementation of the ``OnLineTrainer``, which inherits from ``Trainer``.

"""
__author__ = "Vincent Marois, Tomasz Kornuta"

import numpy as np
from torch.nn.utils import clip_grad_value_

from workers.trainer import Trainer


class OnLineTrainer(Trainer):
    """
    Implementation for the episode-based ``OnLineTrainer``.

    ..note ::

        The ``OfflineTrainer`` is based on epochs. While an epoch can be defined for all finite-size datasets, \
        it makes less sense for problems which have a very large, almost infinite, dataset (like algorithmic \
        tasks, which generate random data on-the-fly). \
         
        This is why this OnLineTrainer was implemented. Instead of looping on epochs, it iterates directly on \
        episodes (we call an iteration on a single batch an episode).


    """

    def __init__(self, name="OnLineTrainer"):
        """
        Only calls the ``Trainer`` constructor as the initialization phase is identical to the ``Trainer``.

       :param name: Name of the worker (DEFAULT: "OnLineTrainer").
       :type name: str

        """ 
        # Call base constructor to set up app state, registry and add default params.
        super(OnLineTrainer, self).__init__(name)

    def setup_experiment(self):
        """
        Sets up experiment for episode trainer:

            - Calls base class setup_experiment to parse the command line arguments,
            - Sets up the terminal conditions (loss threshold, episodes & epochs (optional) limits).

        """
        # Call base method to parse all command line arguments, load configuration, create problems and model etc.
        super(OnLineTrainer, self).setup_experiment()

        ################# TERMINAL CONDITIONS ################# 
        self.logger.info('Terminal conditions:\n' + '='*80)

        # Terminal condition I: loss. 
        self.params['training']['terminal_conditions'].add_default_params({'loss_stop': 1e-5})
        self.loss_stop = self.params['training']['terminal_conditions']['loss_stop']
        self.logger.info("Setting Loss Stop threshold to {}".format(self.loss_stop))

        # In this trainer Partial Validation is mandatory, hence interval must be > 0.
        self.params['validation'].add_default_params({'partial_validation_interval': 100})
        self.partial_validation_interval = self.params['validation']['partial_validation_interval']
        if self.partial_validation_interval <= 0:
            self.logger.error("Episodic Trainer relies on Partial Validation, thus interval must be a positive number!")
            exit(-4)
        else:
            self.logger.info("Partial Validation activated with interval equal to {} episodes".format(self.partial_validation_interval))

        # Terminal condition II: max epochs. Optional.
        self.params["training"]["terminal_conditions"].add_default_params({'epoch_limit': -1})
        self.epoch_limit = self.params["training"]["terminal_conditions"]["epoch_limit"]
        if self.epoch_limit <= 0:
            self.logger.info("Termination based on Epoch Limit is disabled")
            # Set to infinity.
            self.epoch_limit = np.Inf
        else:
            self.logger.info("Setting the Epoch Limit to: {}".format(self.epoch_limit))

        # Calculate the epoch size in terms of episodes.
        epoch_size = self.training_problem.get_epoch_size(self.params["training"]["problem"]["batch_size"])
        self.logger.info('Epoch size in terms of training episodes: {}'.format(epoch_size))

        # Terminal condition III: max episodes. Mandatory.
        self.params["training"]["terminal_conditions"].add_default_params({'episode_limit': 100000})
        self.episode_limit = self.params['training']['terminal_conditions']['episode_limit']
        if self.episode_limit <= 0:
            self.logger.error("OnLine Trainer relies on episodes, thus Episode Limit must be a positive number!")
            exit(-5)
        else:
            self.logger.info("Setting the Episode Limit to: {}".format(self.episode_limit))
        self.logger.info('\n' + '='*80)

    def run_experiment(self):
        """
        Main function of the ``OnLineTrainer``, runs the experiment.

        Iterates over the (cycled) DataLoader (one iteration = one episode).

        .. note::

            The test for terminal conditions (e.g. convergence) is done at the end of each episode. \
            The terminal conditions are as follows:

                - I. The loss is below the specified threshold (using the partial validation loss),
                - TODO: II. Early stopping is set and the full validation loss did not change by delta \
                    for the indicated number of epochs,
                - III. The maximum number of episodes has been met,
                - IV. The maximum number of epochs has been met (OPTIONAL).
            
            Additionally, experiment can be stopped by the user by pressing 'Stop experiment' \
            during visualization.


        The function does the following for each episode:

            - Handles curriculum learning if set,
            - Resets the gradients
            - Forwards pass of the model,
            - Logs statistics and exports to TensorBoard (if set),
            - Computes gradients and update weights
            - Activate visualization if set,
            - Validate the model on a batch according to the validation frequency.
            - Checks the above terminal conditions.


        """
        # Export and log configuration, optionally asking the user for confirmation.
        self.export_experiment_configuration(self.log_dir, "training_configuration.yaml",self.flags.confirm)

        # Initialize TensorBoard and statistics collection.
        self.initialize_statistics_collection()
        self.initialize_tensorboard()

        # cycle the DataLoader -> infinite iterator
        self.training_dataloader = self.cycle(self.training_dataloader)

        try:
            '''
            Main training and validation loop.
            '''
            # Reset the counters.
            episode = 0
            epoch = 0

            # Inform the training problem class that epoch has started.
            self.training_problem.initialize_epoch(epoch)

            # Set default termination cause.
            termination_cause = "Episode limit reached"
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

                # 4.2. Export data to TensorBoard - at logging frequency.
                if (self.training_batch_writer is not None) and (episode % self.flags.logging_interval == 0):
                    self.training_stat_col.export_to_tensorboard()

                    # Export histograms.
                    if self.flags.tensorboard >= 1:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name, param.data.cpu().numpy(), episode,
                                                                         bins='doane')

                            except Exception as e:
                                self.logger.error("  {} :: data :: {}".format(name, e))

                    # Export gradients.
                    if self.flags.tensorboard >= 2:
                        for name, param in self.model.named_parameters():
                            try:
                                self.training_batch_writer.add_histogram(name + '/grad', param.grad.data.cpu().numpy(),
                                                                         episode, bins='doane')

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
                if (episode % self.partial_validation_interval) == 0:

                    # Check visualization flag
                    if 1 <= self.flags.visualize <= 2:
                        self.app_state.visualize = True
                    else:
                        self.app_state.visualize = False

                    # Perform validation.
                    validation_loss = self.validate_on_batch(self.validation_batch, episode, epoch)

                    # Save the model using the latest validation statistics.
                    self.model.save(self.model_dir, self.validation_stat_col)

                    # Terminal conditions.
                    # I. the loss is < threshold (only when curriculum learning is finished if set.)
                    # We check that condition only in validation step!
                    if self.curric_done or not self.must_finish_curriculum:

                        # Check the Partial Validation loss.
                        if (validation_loss < self.loss_stop):
                            termination_cause = "Partial Validation Loss went below Loss Stop " \
                                                "threshold (model converged)."
                            break

                    # II. Early stopping is set and loss hasn't improved by delta in n epochs.
                    # early_stopping(index=epoch, avg_valid_loss). (TODO: coming in next release)
                    # termination_cause = 'Early Stopping.'

                # III. The episodes number limit has been reached.
                if episode+1 >= self.episode_limit:
                    # If we reach this condition, then it is possible that the model didn't converge correctly
                    # but it currently might get better since last validation.

                    if self.validation_stat_col["episode"] != episode:
                        # We still must validate and try to save the model as it may perform better during this episode
                        # (as opposed to the previous II. condition)

                        # Do not visualize.
                        self.app_state.visualize = False
                        self.validate_on_batch(self.validation_batch, episode, epoch)

                        # Save the model.
                        self.model.save(self.model_dir, self.validation_stat_col)

                    termination_cause = "Episode Limit reached."
                    break

                # Check if we are at the end of the 'epoch': indicate that the DataLoader is now cycling.
                if ((episode + 1) % self.training_problem.get_epoch_size(
                        self.params['training']['problem']['batch_size'])) == 0:

                    # Epoch just ended!
                    # Inform the problem class that the epoch has ended.
                    self.training_problem.finalize_epoch(epoch)

                    # Aggregate training statistics for the epoch.
                    self.aggregate_and_export_statistics(self.model, self.training_problem, 
                            self.training_stat_col, self.training_stat_agg, episode, '[Full Training]')

                    # Apply curriculum learning - change some of the Problem parameters
                    self.curric_done = self.training_problem.curriculum_learning_update_params(episode)

                    # IV. Epoch limit has been reached.
                    if epoch+1 >= self.epoch_limit:
                        termination_cause = "Epoch Limit reached"
                        # "Finish" the training.
                        break

                    # Next epoch!
                    epoch += 1
                    self.logger.info('Starting next epoch: {}'.format(epoch))
                    # Inform the training problem class that epoch has started.
                    self.training_problem.initialize_epoch(epoch)

                # Move on to next episode.
                episode += 1

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
            self.validate_on_set(episode, epoch)

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
    trainer = OnLineTrainer()
    # parse args, load configuration and create all required objects.
    trainer.setup_experiment()
    # GO!
    trainer.run_experiment()


if __name__ == '__main__':

    main()
