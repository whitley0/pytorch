#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
author:qingxingmo
datetime: 2022/8/18 15:52
'''

import torch
import torch.nn as nn
import torch.nn.functional as f

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.tensor(alpha)
        self.gamma = gamma

    def forward(self, outputs, targets):
        targets = targets.view(-1, 1)

        logits = f.log_softmax(outputs)
        logits = logits.gather(1, targets).view(-1)

        pt = logits.exp()

        if self.alpha is not None:
            if self.alpha.type() != outputs.type():
                self.alpha = self.alpha.as_type(outputs)
            at = self.alpha.gather(0, targets.view(-1))
            logits = logits * at
        loss = torch.mean(-1 * (1 - pt) ** self.gamma * logits)

        return loss

if __name__ == '__main__':
    x = torch.randn((1, 2))
    y = torch.tensor([0], dtype=torch.int64)
    model = nn.Linear(2, 2)
    o = model(x)
    loss_fnc = FocalLoss(0.5, 2)
    loss = loss_fnc(o, y)
    loss.backward()
    print(model.weight.grad)
