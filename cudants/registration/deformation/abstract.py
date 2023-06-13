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
