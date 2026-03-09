# Copyright (c) 2026 Rohit Jena. All rights reserved.
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


from typing import List, Sequence

from fireants.utils.globals import MIN_IMG_SIZE


def downsample_size(
    size: Sequence[int],
    scale: float,
    min_img_size: int = MIN_IMG_SIZE,
) -> List[int]:
    """Compute spatial size after downsampling by scale, with a minimum size per dimension.

    When scale > 1, each dimension is divided by scale and floored, but not below min_img_size.
    When scale <= 1, the original size is returned unchanged.

    Args:
        size: Spatial dimensions (e.g. [H, W] or [D, H, W]).
        scale: Downsampling factor (e.g. 2 for half resolution).
        min_img_size: Minimum value per dimension when downsampling.

    Returns:
        List of downsampled spatial dimensions.
    """
    if scale > 1:
        return [max(int(s / scale), min_img_size) for s in size]
    return list(size)
