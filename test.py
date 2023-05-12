import torch
from torch.nn.functional import grid_sample, affine_grid
from timeit import timeit, Timer
shape = (1, 1, 150, 240, 240)
affine = torch.eye(3, 4).unsqueeze(0).expand(1, -1, -1).cuda()
grid = affine_grid(affine, shape, align_corners=True).cuda()
mat = torch.randn(1, 2, 3).cuda()
N = 1000
time_gpu = timeit(lambda: torch.einsum('ntd, n...d->n...t', mat, grid), number=N)/N
N = 100
mat = mat.cpu()
grid = grid.cpu()
time_cpu = timeit(lambda: torch.einsum('ntd, n...d->n...t', mat, grid), number=N)/N
print("GPU time: {}, CPU time: {}".format(time_gpu, time_cpu))
print("Speedup: {}x".format(time_cpu/time_gpu))