''' class for SGD for compositive warps '''
import torch
from torch.nn import functional as F
from fireants.utils.imageutils import compute_inverse_warp_displacement
from fireants.utils.imageutils import jacobian as jacobian_fn
from fireants.losses.cc import separable_filtering

class WarpAdam:
    ''' at the moment we only support a single warp function 
    also supports multi-scale (by simply interpolating to the target size)
    shape of warp = [B, H, W, [D], dims]
    '''
    def __init__(self, warp, lr, warpinv=None, beta1=0.9, beta2=0.99, weight_decay=0, eps=1e-8,
                 scaledown=False, multiply_jacobian=False,
                 smoothing_gaussians=None, optimize_inverse_warp=False):
        # init
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        self.n_dims = len(warp.shape) - 2
        # get half resolutions
        self.half_resolution = 1.0/(max(warp.shape[1:-1]) - 1)
        self.warp = warp
        self.warpinv = warpinv
        self.optimize_inverse_warp = optimize_inverse_warp
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_t = 0    # initialize step to 0
        self.weight_decay = weight_decay
        self.multiply_jacobian = multiply_jacobian
        self.scaledown = scaledown   # if true, the scale the gradient even if norm is below 1
        self.exp_avg = torch.zeros_like(warp)
        self.exp_avg_sq = torch.zeros_like(warp)
        self.permute_imgtov = (0, *range(2, self.n_dims+2), 1)  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (0, self.n_dims+1, *range(1, self.n_dims+1))  # [N, dims, HWD] -> [N, HWD, dims]
        # set grid
        self.batch_size = batch_size = warp.shape[0]
        # init grid
        self.affine_init = torch.eye(self.n_dims, self.n_dims+1, device=warp.device)[None].expand(batch_size, -1, -1)
        self.initialize_grid(warp.shape[1:-1])
        # gaussian smoothing parameters (if any)
        self.smoothing_gaussians = smoothing_gaussians
    
    def set_data_and_size(self, warp, size, grid_copy=None, warpinv=None):
        ''' change the optimization variables sizes '''
        self.warp = warp
        mode = 'bilinear' if self.n_dims == 2 else 'trilinear'
        self.exp_avg = F.interpolate(self.exp_avg.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                            ).permute(*self.permute_imgtov)
        self.exp_avg_sq = F.interpolate(self.exp_avg_sq.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                            ).permute(*self.permute_imgtov)
        self.half_resolution = 1.0/(max(warp.shape[1:-1]) - 1)
        self.initialize_grid(size, grid_copy=grid_copy)
        # print(self.warp.shape, warpinv)
        if self.optimize_inverse_warp and warpinv is not None:
            self.warpinv = warpinv
    
    def initialize_grid(self, size, grid_copy=None):
        ''' initialize the grid (so that we can use it independent of the grid elsewhere) '''
        if grid_copy is None:
            self.grid = F.affine_grid(self.affine_init, [self.batch_size, 1, *size], align_corners=True).detach()
        else:
            self.grid = grid_copy 

    def zero_grad(self):
        ''' set the gradient to none '''
        self.warp.grad = None
    
    def augment_jacobian(self, u):
        # Multiply u (which represents dL/dphi most likely) with Jacobian indexed by J[..., xyz, ..., phi]
        jac = jacobian_fn(self.warp.data + self.grid, normalize=True)  # [B, dims, H, W, [D], dims]
        if self.n_dims == 2:
            ujac = torch.einsum('bxhwp,bhwp->bhwx', jac, u)
        else:
            ujac = torch.einsum('bxhwdp,bhwdp->bhwdx', jac, u)
        return ujac
    
    def step(self):
        ''' check for momentum, and other things '''
        grad = torch.clone(self.warp.grad.data).detach()
        if self.multiply_jacobian:
            grad = self.augment_jacobian(grad)
        # add weight decay term
        if self.weight_decay > 0:
            grad.add_(self.warp.data, alpha=self.weight_decay)
        # compute moments
        self.step_t += 1
        self.exp_avg.mul_(self.beta1).add_(grad, alpha=1-self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad.conj(), value=1-self.beta2)
        # bias correction
        beta_correction1 = 1 - self.beta1 ** self.step_t
        beta_correction2 = 1 - self.beta2 ** self.step_t
        denom = (self.exp_avg_sq / beta_correction2).sqrt().add_(self.eps)
        # get updated gradient (this will be normalized and passed in)
        grad = self.exp_avg / beta_correction1 / denom
        # renormalize and update warp
        # gradmax = self.eps + grad.reshape(grad.shape[0], -1).abs().max(1).values  # [B,]
        gradmax = self.eps + grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
        gradmax = gradmax.reshape(-1, *([1])*(self.n_dims+1))
        if not self.scaledown:  # if scaledown is "True", then we scale down even if the norm is below 1
            gradmax = torch.clamp(gradmax, min=1)
        # print(gradmax.abs().min(), gradmax.abs().max())
        grad = grad / gradmax * self.half_resolution   # norm is now 0.5r
        # multiply by learning rate
        grad.mul_(-self.lr)
        # print(grad.abs().max().item(), self.half_resolution, self.warp.shape)
        # compositional update
        w = grad + F.grid_sample(self.warp.data.permute(*self.permute_vtoimg), self.grid + grad, mode='bilinear', align_corners=True).permute(*self.permute_imgtov)
        # w = grad + self.warp.data
        # smooth result if asked for
        if self.smoothing_gaussians is not None:
            w = separable_filtering(w.permute(*self.permute_vtoimg), self.smoothing_gaussians).permute(*self.permute_imgtov)
        self.warp.data.copy_(w)
        # add to inverse if exists
        if self.optimize_inverse_warp and self.warpinv is not None:
            invwarp = compute_inverse_warp_displacement(self.warp.data, self.grid, self.warpinv.data, iters=5)
            warp_new = compute_inverse_warp_displacement(invwarp, self.grid, self.warp.data, iters=5) 
            self.warp.data.copy_(warp_new)
            self.warpinv.data.copy_(invwarp)
