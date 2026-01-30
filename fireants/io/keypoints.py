''' data structure and utilities to handle keypoints ''' 
import torch
from fireants.io.image import Image
from logging import getLogger
import logging
from fireants.types import devicetype
from typing import Union, List
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Keypoints:
    def __init__(self, 
            keypoints: Union[torch.Tensor, np.ndarray], 
            image: Image,
            device: devicetype = 'cuda',
            space: str = 'pixel',
    ):
        """
        Keypoints stores a set of N keypoint coordinates for a given image, along with associated metadata.

        The keypoints can be represented in one of three coordinate spaces:
        - 'pixel': (x, y, [z]) indices in image pixel space, where (0,0,[0]) refers to the first pixel/voxel.
        - 'physical': world coordinates corresponding to real-space units (e.g., millimeters), using image origin, spacing, and direction.
        - 'torch': torch (PyTorch) coordinate space, typically matching SimpleITK physical space but may have different axis order.

        Parameters
        ----------
        keypoints : (N, dims) torch.Tensor or np.ndarray
            The keypoint locations as a 2D tensor/array, one row per keypoint, columns for spatial dimensions.
        image : Image
            Reference Image object, provides metadata for coordinate transforms.
        device : str or torch.device, optional
            Device on which to store the keypoints tensor. Default: 'cuda'.
        space : str, optional
            Coordinate space for stored keypoints ('pixel', 'physical', or 'torch'). Default: 'pixel'.

        Attributes
        ----------
        keypoints : torch.Tensor
            The underlying keypoints tensor of shape (N, dims).
        dims : int
            Number of spatial dimensions (2 for 2D, 3 for 3D).
        space : str
            The coordinate space in which keypoints are currently represented.
        device : str or torch.device
            Device on which the keypoints tensor is stored.
        num_keypoints : int
            Number of keypoints stored.

        Provides utilities for updating, transforming, and converting keypoints between spaces.

        Raises
        ------
        AssertionError
            If `space` is not one of 'pixel', 'physical', or 'torch'.
            If `keypoints` tensor is not 2D or does not match image dimensionality.
        """

        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).float().contiguous()
        self.keypoints = keypoints.to(device).float().contiguous()
        self.dims = image.dims
        assert space in ['pixel', 'physical', 'torch'], "Invalid space (only pixel, physical, and torch are supported)"
        assert self.keypoints.ndim == 2, "Keypoints must be a 2D tensor"
        assert self.keypoints.shape[1] == self.dims, "Keypoints must have the same dimensionality as the image"

        # borrow from image
        self.phy2torch = image.phy2torch
        self.torch2phy = image.torch2phy
        # compute as torch tensors
        self.px2phy = torch.from_numpy(image._px2phy).to(device).float().unsqueeze_(0)
        self.phy2px = torch.inverse(self.px2phy[0]).float().unsqueeze_(0)
        self.torch2px = torch.from_numpy(image._torch2px).to(device).float().unsqueeze_(0)
        self.px2torch = torch.inverse(self.torch2px[0]).float().unsqueeze_(0)
        self.dims = image.dims
        # space in which keypoints as stored
        self.space = space
        self.device = device
        self.num_keypoints = keypoints.shape[0]
    
    def update_keypoints(self, keypoints: Union[torch.Tensor, np.ndarray]):
        ''' function to update the actual keypoints in the object '''
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints)
        assert keypoints.ndim == 2, "Keypoints must be a 2D tensor"
        assert keypoints.shape[1] == self.dims, "Keypoints must have the same dimensionality as the image"
        self.keypoints = keypoints.to(self.device).float().contiguous()
        self.num_keypoints = keypoints.shape[0]
        return self
    
    def update_space(self, space: str):
        ''' function to update the space in which the keypoints are stored '''
        assert space in ['pixel', 'physical', 'torch'], "Invalid space (only pixel, physical, and torch are supported)"
        self.space = space
        return self

    def to(self, device: devicetype):
        self.keypoints = self.keypoints.to(device)
        self.device = device
        return self
    
    def _transform_space(self, matrix):
        ''' 
        given a matrix, transform the keypoints to the new space given by the matrix 
        matrix is assumed to be a (n+1, n+1) matrix
        '''
        ret = torch.einsum('ij, nj->ni', matrix[:self.dims, :self.dims], self.keypoints) + matrix[:self.dims, -1]
        ret = ret.contiguous()
        return ret

    def as_pixel_coordinates(self):
        # return keypoints in pixel coordinates
        if self.space == 'pixel':
            return self.keypoints
        elif self.space == 'physical':
            return self._transform_space(self.phy2px[0])
        elif self.space == 'torch':
            return self._transform_space(self.torch2px[0])
        
    def as_physical_coordinates(self):
        # return keypoints in physical coordinates
        if self.space == 'pixel':
            return self._transform_space(self.px2phy[0])
        elif self.space == 'physical':
            return self.keypoints
        elif self.space == 'torch':
            return self._transform_space(self.torch2phy[0])

    def as_torch_coordinates(self):
        # return keypoints in torch coordinates
        if self.space == 'pixel':
            return self._transform_space(self.px2torch[0])
        elif self.space == 'physical':
            return self._transform_space(self.phy2torch[0])
        elif self.space == 'torch':
            return self.keypoints


class BatchedKeypoints:
    ''' the counterpart to BatchedImages for Keypoints '''
    def __init__(self, keypoints: Union[Keypoints, List[Keypoints]], ):
        if isinstance(keypoints, Keypoints):
            keypoints = [keypoints]
        num_keypoints = [x.num_keypoints for x in keypoints]
        self.can_collate = all([x == num_keypoints[0] for x in num_keypoints])  # if all keypoints have the same number of keypoints, then we can collate, else not
        self.keypoints = keypoints
    
    def __len__(self):
        return len(self.keypoints)
    
    def as_torch_coordinates(self):
        tensors = [x.as_torch_coordinates() for x in self.keypoints]
        if self.can_collate:
            return torch.cat([x.unsqueeze(0) for x in tensors], dim=0)
        else:
            return tensors
    
    def as_physical_coordinates(self):
        tensors = [x.as_physical_coordinates() for x in self.keypoints]
        if self.can_collate:
            return torch.cat([x.unsqueeze(0) for x in tensors], dim=0)
        else:
            return tensors
    
    def as_pixel_coordinates(self):
        tensors = [x.as_pixel_coordinates() for x in self.keypoints]
        if self.can_collate:
            return torch.cat([x.unsqueeze(0) for x in tensors], dim=0)
        else:
            return tensors
    
    @staticmethod
    def transform_keypoints_batch(matrix: torch.Tensor, keypoints_batch: torch.Tensor):
        ''' transform keypoints by a matrix '''
        dims = keypoints_batch.shape[-1]
        ret = torch.einsum('bij, bnj->bni', matrix[:, :dims, :dims], keypoints_batch) + matrix[:, :dims, -1]
        ret = ret.contiguous()
        return ret
    
    @classmethod
    def from_tensor_and_metadata(cls, tensor: Union[torch.Tensor, List[torch.Tensor]], batched_keypoints: 'BatchedKeypoints', space: str = 'torch'):
        ''' create a new batched keypoints object from a tensor and metadata '''
        if isinstance(tensor, torch.Tensor):
            tensor = [tensor[i] for i in range(tensor.shape[0])]
        assert len(tensor) == len(batched_keypoints), "Number of tensors must match the number of keypoints"
        new_kps = []
        for i, kps in enumerate(batched_keypoints.keypoints):
            nval = deepcopy(kps)
            nval.update_keypoints(tensor[i].to(nval.device))
            nval.update_space(space)
            new_kps.append(nval)
        return cls(new_kps)

# utility functions
def _compute_keypoint_distance_helper(keypoints1: Keypoints, keypoints2: Keypoints, space: str = 'pixel', reduction: str = 'mean'):
    ''' compute the distance between two keypoints '''
    if space == 'pixel':
        kp1 = keypoints1.as_pixel_coordinates()
        kp2 = keypoints2.as_pixel_coordinates()
    elif space == 'physical':
        kp1 = keypoints1.as_physical_coordinates()
        kp2 = keypoints2.as_physical_coordinates()
    elif space == 'torch':
        kp1 = keypoints1.as_torch_coordinates()
        kp2 = keypoints2.as_torch_coordinates()
    ret = torch.norm(kp1 - kp2, dim=-1)
    if reduction == 'mean':
        return ret.mean()
    elif reduction == 'sum':
        return ret.sum()
    elif reduction == 'none' or reduction is None:
        return ret
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
def compute_keypoint_distance(keypoints1: Union[Keypoints, BatchedKeypoints], keypoints2: Union[Keypoints, BatchedKeypoints], space: str = 'pixel', reduction: str = 'mean'):
    ''' compute the distance between two keypoints '''
    if isinstance(keypoints1, BatchedKeypoints):
        keypoints1 = keypoints1.keypoints # list of keypoints
    else:
        keypoints1 = [keypoints1]
    if isinstance(keypoints2, BatchedKeypoints):
        keypoints2 = keypoints2.keypoints # list of keypoints
    else:
        keypoints2 = [keypoints2]
    assert len(keypoints1) == len(keypoints2), "Number of keypoints must match"
    ret = []
    for kp1, kp2 in zip(keypoints1, keypoints2):
        ret.append(_compute_keypoint_distance_helper(kp1, kp2, space, reduction))
    return torch.stack(ret)
