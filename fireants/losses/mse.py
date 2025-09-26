# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


'''
Cross correlation
'''
from time import time, sleep
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable

class NoOp(nn.Module):
    ''' dummy loss function that does not penalize anything 

    this can be used for regularization only. 
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ignore everything
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0).to(pred.dtype).to(pred.device)
    
    def get_image_padding(self) -> int:
        return 0
        

class MeanSquaredError(nn.Module):
    """
    """

    def __init__(
        self,
        reduction: str = "mean",
    ) -> None:
        """
       """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        mse = F.mse_loss(pred, target, reduction=self.reduction)
        return mse
    
    def get_image_padding(self) -> int:
        return 0


if __name__ == '__main__':
    N = 64  
    img1 = torch.rand(1, 1, N, N, N).cuda()
    img2 = torch.rand(1, 1, N, N, N).cuda()
    # loss = torch.jit.script(LocalNormalizedCrossCorrelationLoss(3, kernel_type='rectangular', reduction='mean')).cuda()
    loss = MeanSquaredError().cuda()
    total = 0
    @torch.jit.script
    def train(img1: torch.Tensor, img2: torch.Tensor, n: int) -> float:
        total = 0.0
        for i in range(n):
            out = loss(img1, img2)
            total += out.item()
        return total
    
    a = time()
    # total = train(img1, img2, 200)
    for i in range(200):
        out = loss(img1, img2)
        total += out.item()
    print(time() - a)
    print('mse', total / 200)