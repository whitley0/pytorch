#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
author:qingxingmo
datetime: 2022/8/7 10:45
'''
import copy

import torch.nn as nn

if __name__ == '__main__':
    model = nn.Conv2d(3, 4, 3).to('cuda')
    model1 = copy.deepcopy(model)
    print(model.weight.device)
    print(model1.weight.device)
