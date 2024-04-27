# Copyright (c) 2021 torchtorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

"""
Warp the functon api, so the normal and quantization training can use the same network.
"""


class Add(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return torch.add(x, y, name)


class Subtract(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return torch.subtract(x, y, name)


class Multiply(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return torch.multiply(x, y, name)


class Divide(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, name=None):
        return torch.divide(x, y, name)


class Reshape(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, shape, name=None):
        return torch.reshape(x, shape, name)


class Transpose(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, perm, name=None):
        return torch.transpose(x, perm, name)


class Concat(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis=0, name=None):
        return torch.concat(x, axis, name)


class Flatten(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, start_axis=0, stop_axis=-1, name=None):
        return torch.flatten(x, start_axis, stop_axis, name)
