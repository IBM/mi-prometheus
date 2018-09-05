#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""param_gen.py: The class that converts the hidden vector of the controller into the parameters of the interface
"""
__author__ = " Ryan L. McAvoy"

from torch import nn
import torch.nn.functional as F



class Param_Generator(nn.Module):
    def __init__(self,
               param_in_dim,
         #      memory_size=128,
               word_size=20,
               num_reads=1,
               num_writes=1,
               shift_size=3):

        """Initialize all the parameters of the interface

        :param param_in_dim: input size. (typically the size of the hidden state)
        :param word_size: size of the word in memory 
        :param num_reads: number of read heads
        :param num_writes: number of write heads
        :param shift_size: size of the shift vector (3 means it can go forward, backward and remain in place)
        """
        super(Param_Generator, self).__init__()


        #self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._num_shifts = shift_size
       
        # v_t^i - The vectors to write to memory, for each write head `i`.
        self.write_vect_ = nn.Linear(param_in_dim, self._num_writes*self._word_size)
        
        # e_t^i - Amount to erase the memory by before writing, for each write head.
        self.erase_vect_ = nn.Linear(param_in_dim, self._num_writes*self._word_size)
       
        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        self.free_gate_ = nn.Linear(param_in_dim, self._num_reads)
      
        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        self.allocate_gate_ = nn.Linear(param_in_dim, self._num_writes)
      
        # g_t^{w, i} - Overall gating of write amount for each write head.
        self.write_gate_ = nn.Linear(param_in_dim, self._num_writes)

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        num_read_modes=1+2 * self._num_writes
        self.read_mode_ = nn.Linear(param_in_dim, self._num_reads*num_read_modes)
        
        # Parameters for the (read / write) "weights by content matching" modules.
        self.write_keys_ = nn.Linear(param_in_dim, self._num_writes*self._word_size)
        self.write_strengths_ = nn.Linear(param_in_dim, self._num_writes)

        self.read_keys_ = nn.Linear(param_in_dim, self._num_reads*self._word_size)
        self.read_strengths_ = nn.Linear(param_in_dim, self._num_reads)
      
 
        #s_j The shift vector that defines the circular convolution of the outputs
        self.shifts_ = nn.Linear(param_in_dim, self._num_shifts*self._num_writes)
        
        # \gamma, sharpening parameter for the weights
        self.sharpening_ = nn.Linear(param_in_dim, self._num_writes)
        self.sharpening_r_ = nn.Linear(param_in_dim, self._num_reads) 
        self.shifts_r_ = nn.Linear(param_in_dim, self._num_shifts*self._num_reads)


    def forward(self, vals):
        """
        Calculates the controller parameters
        
        :param vals: data from the controller (from time t). Typically, the hidden state.  [BATCH_SIZE x INPUT_SIZE]
        :return update_data: dictionary (update_data contains all of the controller parameters)
        """
        
        update_data={}


        # v_t^i - The vectors to write to memory, for each write head `i`.
        update_data['write_vectors'] = self.write_vect_(vals).view(-1,self._num_writes,self._word_size)
        
        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vec=F.sigmoid(self.erase_vect_(vals))  # [batch, num_writes*word_size]
        update_data['erase_vectors'] = erase_vec.view(-1,self._num_writes,self._word_size)
        
        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        update_data['free_gate'] = F.sigmoid(self.free_gate_(vals)).view(-1,self._num_reads,1)
        
        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        update_data['allocation_gate'] = F.sigmoid(self.allocate_gate_(vals)).view(-1,self._num_writes,1)

        # g_t^{w, i} - Overall gating of write amount for each write head.
        update_data['write_gate'] = F.sigmoid(self.write_gate_(vals)).view(-1,self._num_writes,1)
        
        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        #Need to apply softmax batch-wise to the second index. This will not work
        num_read_modes=1+2 * self._num_writes
        read_mode=F.softmax(self.read_mode_(vals),-1)
        update_data['read_mode'] = read_mode.view(-1,self._num_reads,num_read_modes)
        
        # Parameters for the (read / write) "weights by content matching" modules.
        update_data['write_content_keys'] = self.write_keys_(vals).view(-1,self._num_writes,self._word_size)
        update_data['write_content_strengths'] = 1 + F.softplus(self.write_strengths_(vals)).view(-1,self._num_writes,1)
        
        update_data['read_content_keys'] = self.read_keys_(vals).view(-1,self._num_reads,self._word_size)
        update_data['read_content_strengths'] = 1 + F.softplus(self.read_strengths_(vals)).view(-1,self._num_reads,1) 
       

        #s_j The shift vector that defines the circular convolution of the outputs
        shifts = F.softmax(F.softplus(self.shifts_(vals)),dim=-1)
        update_data['shifts'] = shifts.view(-1,self._num_writes,self._num_shifts)
         
        shifts_r = F.softmax(F.softplus(self.shifts_r_(vals)),dim=-1)
        update_data['shifts_read'] = shifts_r.view(-1,self._num_reads,self._num_shifts)
        
        # \gamma, sharpening parameter for the weights
        update_data['sharpening'] =1+F.softplus(self.sharpening_(vals)).view(-1,self._num_writes,1)
        update_data['sharpening_read'] =1+F.softplus(self.sharpening_r_(vals)).view(-1,self._num_reads,1)
          
        update_data['read_mode_shift'] = F.sigmoid(self.free_gate_(vals)).view(-1,self._num_reads,1)

        return update_data


