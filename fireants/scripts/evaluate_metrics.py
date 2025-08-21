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


from collections import defaultdict
import numpy as np

def compute_metrics(fixed_array, moved_array, eps=0):
    ''' given the fixed and moved arrays, compute the metrics and return a dictionary of metrics 
    Here the fixed array and moved array are both numpy arrays of size [C, H, W, D] 
    '''
    metrics = defaultdict(list)
    # compute scores
    int_all, diff_s_t = 0, 0
    num_s_all, num_t_all = 0, 0
    num_s_minus_t_all, num_t_minus_s_all = 0, 0

    for lab in range(fixed_array.shape[0]):
        # compute 
        s = (moved_array[lab]).astype(np.float32)  # segmentation
        t = (fixed_array[lab]).astype(np.float32)  # target 
        intersection = (s*t).sum()
        num_s = s.sum() + eps
        num_t = t.sum() + eps
        s_minus_t = (s*(1-t)).sum()
        t_minus_s = (t*(1-s)).sum()
        metrics['target_overlap'].append(intersection/num_t)
        metrics['mean_overlap'].append(2*intersection/(num_s + num_t))
        metrics['false_negatives'].append(t_minus_s/num_t)
        metrics['false_positives'].append(s_minus_t/num_s)
        metrics['volume_sim'].append(2*(abs(num_s - num_t))/(num_s + num_t))
        # add it to all 
        int_all += intersection
        num_s_all += num_s
        num_t_all += num_t
        num_s_minus_t_all += s_minus_t
        num_t_minus_s_all += t_minus_s
        diff_s_t += abs(num_s - num_t)

    # compute klein metrics
    metrics['target_overlap_klein'] = int_all/num_t_all
    metrics['mean_overlap_klein'] = 2*int_all/(num_s_all + num_t_all)
    metrics['false_negatives_klein'] = num_t_minus_s_all/num_t_all
    metrics['false_positives_klein'] = num_s_minus_t_all/num_s_all
    metrics['volume_sim_klein'] = 2*diff_s_t/(num_s_all + num_t_all)

    return metrics
