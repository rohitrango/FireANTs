'''
Cross correlation
'''
from time import time, sleep
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, List, Optional, Dict, Any, Callable

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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        mse = F.mse_loss(pred, target, reduction=self.reduction)
        return mse


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