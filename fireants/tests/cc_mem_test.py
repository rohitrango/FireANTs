from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList

if __name__ == "__main__":
    torch.cuda.memory._record_memory_history()
    size = (2, 4, 128, 128, 128)
    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    print("Without checkpointing")
    loss_fn = LocalNormalizedCrossCorrelationLoss(reduction="mean")
    start = time()
    loss = loss_fn(img, img2)
    loss.backward()
    print(loss.item())
    print(time() - start)
    del loss, loss_fn, img, img2
    torch.cuda.empty_cache()
    sleep(1)
    img = torch.randn(*size).cuda()
    img2 = torch.randn(*size).cuda().requires_grad_(True)
    print("With checkpointing")
    loss_fn = LocalNormalizedCrossCorrelationLoss(reduction="mean", checkpointing=True)
    start = time()
    loss = loss_fn(img, img2)
    loss.backward()
    print(loss.item())
    print(time() - start)
    del loss, loss_fn, img, img2
    torch.cuda.memory._dump_snapshot("cc_mem_test.pkl")
