'''
Methods for implementing optical flow algorithms (used by logDemon variants).
'''
import torch
from torch import nn
from cudants.utils.imageutils import image_gradient_nograd, image_gradient 
from cudants.types import devicetype
from cudants.losses.cc import separable_filtering, gaussian_1d
from cudants.utils.util import catchtime

class OpticalFlow(nn.Module):
    '''
    sigma: in pixels
    TODO fill this out
    '''
    def __init__(self, method: str = 'thirions', 
                 sigma: float = 2.0, no_grad=True, 
                 eps: float = 1e-3,
                 device: devicetype = 'cuda') -> None:
        super().__init__()
        self._method_name = method
        if method == 'thirions':
            self._method = self._thirions 
        else:
            raise ValueError(f"Optical flow method {method} not supported yet.")
        # initialize smoothing and gradients
        self.gradient_kernel = image_gradient
        self.gaussian_kernel = gaussian_1d(torch.tensor(sigma, device=device))
        self.no_grad = no_grad
        self.grad_context = torch.set_grad_enabled(not self.no_grad)
        self.eps = eps
    
    def __str__(self) -> str:
        return "OpticalFlow: using {} backend with gaussian kernel of size {}".format(self._method_name, self.gaussian_kernel.shape)
    
    def _thirions(self, I: torch.Tensor, J:torch.Tensor):
        ''' thirions formula for optical flow 
        I is considered the fixed image, J is the moving image
        '''
        gradJ = self.gradient_kernel(J) 
        gradJnorm_squared = (gradJ**2).sum(dim=1, keepdim=True)
        diff = I - J
        denom = gradJnorm_squared + diff**2
        # flow = torch.where(denom > self.eps, diff * gradJ / denom, 0) 
        flow = diff * gradJ / denom * (denom > self.eps).float()
        flow = separable_filtering(flow, self.gaussian_kernel)
        return flow
    
    def forward(self, I: torch.Tensor, J: torch.Tensor):
        with self.grad_context:
            return self._method(I, J)


if __name__ == '__main__':
    opticalflow = OpticalFlow(sigma=1.0)
    N = 64
    I = torch.rand(1, 1, N, N, N, device='cuda')
    J = torch.rand(1, 1, N, N, N, device='cuda')
    print(opticalflow)
    from time import time
    a = time()
    out = opticalflow(I, J)
    print(time() - a)
