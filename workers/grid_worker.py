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
grid_worker.py:

    - This file contains the base parser used by every worker. Having a root common parser\
     shared by all workers allow easier future maintenance, as it eases adding an argument\
      to all workers. Each worker is able to add specific arguments using the ``add_arguments()``\
       function defined in the relevant module.

    - The base worker class is also defined here.


"""
__author__ = "Vincent Marois, Tomasz Kornuta"