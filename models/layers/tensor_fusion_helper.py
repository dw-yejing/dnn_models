# Copyright (c) 2022 torchtorch Authors. All Rights Reserved.
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
import torch.nn.functional as F


def avg_reduce_hw(x):
    # Reduce hw by avg
    # Return cat([avg_pool_0, avg_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return F.adaptive_avg_pool2d(x, 1)
    elif len(x) == 1:
        return F.adaptive_avg_pool2d(x[0], 1)
    else:
        res = []
        for xi in x:
            res.append(F.adaptive_avg_pool2d(xi, 1))
        return torch.concat(res, axis=1)


def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when axis=[2, 3], the torch.max api has bug for training.
    if is_training:
        max_pool = F.adaptive_max_pool2d(x, 1)
    else:
        max_pool = torch.max(x, axis=[2, 3], keepdim=True)

    if use_concat:
        res = torch.concat([avg_pool, max_pool], axis=1)
    else:
        res = [avg_pool, max_pool]
    return res


def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.concat(res, axis=1)


def avg_reduce_channel(x):
    # Reduce channel by avg
    # Return cat([avg_ch_0, avg_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.mean(x, axis=1, keepdim=True)
    elif len(x) == 1:
        return torch.mean(x[0], axis=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.mean(xi, axis=1, keepdim=True))
        return torch.concat(res, axis=1)


def max_reduce_channel(x):
    # Reduce channel by max
    # Return cat([max_ch_0, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return torch.max(x, axis=1, keepdim=True)
    elif len(x) == 1:
        return torch.max(x[0], axis=1, keepdim=True)
    else:
        res = []
        for xi in x:
            res.append(torch.max(xi, axis=1, keepdim=True))
        return torch.concat(res, axis=1)


def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, axis=1, keepdim=True)
    max_value = torch.max(x, axis=1, keepdim=True)

    if use_concat:
        res = torch.concat([mean_value, max_value], axis=1)
    else:
        res = [mean_value, max_value]
    return res


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.concat(res, axis=1)


def cat_avg_max_reduce_channel(x):
    # Reduce hw by cat+avg+max
    assert isinstance(x, (list, tuple)) and len(x) > 1

    x = torch.concat(x, axis=1)

    mean_value = torch.mean(x, axis=1, keepdim=True)
    max_value = torch.max(x, axis=1, keepdim=True)
    res = torch.concat([mean_value, max_value], axis=1)

    return res