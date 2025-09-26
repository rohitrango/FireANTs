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


from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from fireants.losses.cc import gaussian_1d
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
from fireants.losses.cc import separable_filtering, _separable_filtering_conv

jitted = torch.jit.script(separable_filtering)

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history()
    size = (2, 4, 128, 128, 128)
    img = torch.randn(*size).cuda()
    kernels = [gaussian_1d(s, truncated=2) for s in torch.ones(3, device='cuda')]
    print([k for k in kernels[:1]])
    img = separable_filtering(img, kernels)
    del img, kernels
    torch.cuda.empty_cache()

    img = torch.randn(*size).cuda()
    kernels = [gaussian_1d(s, truncated=2) for s in torch.ones(3, device='cuda')]
    img = jitted(img, kernels)
    del img, kernels
    torch.cuda.empty_cache()
    torch.cuda.memory._dump_snapshot("conv_mem_test.pkl")
