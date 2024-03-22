'''
Methods for implementing optical flow algorithms (used by logDemon variants).
'''
import torch
from torch import nn
from fireants.utils.imageutils import image_gradient_nograd, image_gradient 
from fireants.types import devicetype
from fireants.losses.cc import separable_filtering, gaussian_1d
from fireants.utils.util import catchtime

class OpticalFlow(nn.Module):
    '''
    sigma: in pixels
    TODO fill this out
    '''
    def __init__(self, method: str = 'gauss-newton', 
                 sigma: float = 2.0, no_grad=True, 
                 eps: float = 1e-3,
                 device: devicetype = 'cuda') -> None:
        super().__init__()
        self._method_name = method
        if method == 'thirions':
            self._method = self._thirions 
        elif method == 'esm':
            self._method = self._esm
        elif method == 'gauss-newton':
            self._method = self._gaussnewton
        elif method == 'grad-msd':
            self._method = self._grad_msd
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

    def _grad_msd(self, I: torch.Tensor, J: torch.Tensor):
        Jp = self.gradient_kernel(J)
        Jpnorm_squared = 1 - (I - J)**2
        return Jp, Jpnorm_squared
    
    def _thirions(self, I: torch.Tensor, J:torch.Tensor):
        ''' thirions formula for optical flow 
        I is considered the fixed image, J is the moving image (current iteration)
        '''
        Jp = self.gradient_kernel(I)
        Jpnorm_squared = (Jp**2).sum(dim=1, keepdim=True)
        return Jp, Jpnorm_squared
    
    def _esm(self, I: torch.Tensor, J: torch.Tensor):
        ''' Efficient second order minimization (ESM) method for optical flow '''
        Jp = 0.5*(self.gradient_kernel(J) + self.gradient_kernel(I))
        Jpnorm_squared = (Jp**2).sum(dim=1, keepdim=True)
        return Jp, Jpnorm_squared
    
    def _gaussnewton(self, I: torch.Tensor, J: torch.Tensor):
        ''' Gauss-Newton method for optical flow '''
        Jp = self.gradient_kernel(J)
        Jpnorm_squared = (Jp**2).sum(dim=1, keepdim=True)
        return Jp, Jpnorm_squared
    
    def forward(self, I: torch.Tensor, J: torch.Tensor):
        ''' 
        I: fixed image, J: moving image of size [N, 1, ...] 
        returns flow of size [N, dims, ...]
        '''
        with self.grad_context:
            diff = I - J
            # refer to https://www.insight-journal.org/browse/publication/644 for different methods
            Jp, Jpnorm_squared = self._method(I, J)
            denom = Jpnorm_squared + diff**2
            flow = torch.where(denom >= self.eps, diff * Jp / denom, 0)
            flow = separable_filtering(flow, self.gaussian_kernel)
            return flow


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
