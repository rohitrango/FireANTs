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
author: rohitrango

Abstract deformation class
'''
from abc import ABC, abstractmethod

class AbstractDeformation(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_warp(self):
        ''' returns displacement field '''
        raise NotImplementedError
    
    @abstractmethod
    def get_inverse_warp(self):
        ''' returns inverse displacement field '''
        raise NotImplementedError 
    
    @abstractmethod
    def set_size(self, size):
        ''' sets size of the deformation field '''
        raise NotImplementedError
    
    @abstractmethod
    def step(self):
        ''' optimizes the deformation field '''
        raise NotImplementedError
