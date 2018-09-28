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


from memory_usage import MemoryUsage
# Tests for MemoryUsage
leng = 10
mem_use = MemoryUsage(1)

usage = mem_use.init_state(10, 3)
print(usage.shape)
print(usage)
usage[0, 0] = 20
usage[2, 0] = 2
usage[0, 1:] = 3
usage[2, 2:] = 4
usage[0, leng - 1] = 49494994
print(usage.shape)
print(usage)
cumprod = mem_use.exclusive_cumprod_temp(usage)
print(cumprod.shape)
print(cumprod)
