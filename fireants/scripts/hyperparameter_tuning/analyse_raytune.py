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
Script to visualize and plot the hyperparameter tuning results from ray.tune
'''
import numpy as np
import json
import argparse 
from glob import glob
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--tune_dir', type=str, required=True, help='path to directory containing all the ')


if __name__ == '__main__':
    args = parser.parse_args()
    tune_dir = args.tune_dir

    # get all the results
    all_results = []

    dirs = glob(os.path.join(tune_dir, '*'))
    for dir in tqdm(dirs):
        if os.path.isdir(dir):
            result_file = os.path.join(dir, 'result.json')
            with open(result_file, 'r') as f:
                data = f.readlines()
                data = [json.loads(line) for line in data]
                data = data[-1]
                for k, v in data['config'].items():
                    data[k] = v
            all_results.append(data)
        else:
            continue
    # create dataframe
    dataframe = pd.DataFrame(all_results)
    dataframe = dataframe.sort_values(by=['target_overlap'], ascending=False)
    dataframe.reset_index(inplace=True)
    # print(dataframe)
    print("Best results...")
    for i in range(20):
        print(dataframe.loc[i, 'target_overlap'], dataframe.loc[i, 'config'])
    print("\nWorst results...")
    N = len(dataframe)
    for i in range(20):
        print(dataframe.loc[N-20+i, 'target_overlap'], dataframe.loc[N-20+i, 'config'])
    # print dataframe keys
    print("Printing keys of the dataframe...")
    print(dataframe.keys())
    