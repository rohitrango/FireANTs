'''
This module provides classes for handling medical images with SimpleITK backend and PyTorch tensor support.
It includes functionality for loading, manipulating, and transforming images, as well as batching operations for efficiency.
'''

import torch
import SimpleITK as sitk
import numpy as np
from typing import Any, Union, List, Tuple
from time import time
from fireants.types import devicetype
from fireants.utils.imageutils import integer_to_onehot

class Image:
    '''`Image` is a class to handle medical images with SimpleITK backend and PyTorch tensor support.

    This class provides functionality to work with 2D/3D medical images, handling both regular images
    and segmentation masks. It maintains both SimpleITK and PyTorch representations, and handles
    coordinate transformations between physical, pixel, and normalized coordinate spaces.

    Args:
        itk_image (SimpleITK.Image): The SimpleITK image object
        device (devicetype, optional): Device to store the PyTorch tensors on. Defaults to 'cuda'.
        is_segmentation (bool, optional): Whether the image is a segmentation mask. Defaults to False.
            Set this to True if the image is an integer-valued segmentation mask.
            This is used to convert integer labels to one-hot encoding.
            See `max_seg_label`, `background_seg_label`, `seg_preprocessor` for more details on how to manipulate integer label images.
        max_seg_label (int, optional): Maximum label value for segmentation. Values above this are set to background_seg_label.
            Set to None by default, meaning no label clipping is done.
        background_seg_label (int, optional): Label value representing background in segmentations. Defaults to 0.
        seg_preprocessor (callable, optional): Function to preprocess segmentation arrays. Defaults to identity function.
        spacing (array-like, optional): Custom spacing for the image. If None, uses SimpleITK values.
        direction (array-like, optional): Custom direction matrix. If None, uses SimpleITK values.
        origin (array-like, optional): Custom origin point. If None, uses SimpleITK values.
        center (array-like, optional): Custom center point to recalibrate origin. If provided, origin is recalculated.

    Attributes:
        array (torch.Tensor): PyTorch tensor representation of the image
            This is of size [1, C, *dims] where C is the number of channels / labels
        itk_image (SimpleITK.Image): Original SimpleITK image
        channels (int): Number of channels in the image
        dims (int): Dimensionality of the image (2 or 3)
        torch2phy (torch.Tensor): Transform matrix from normalized to physical coordinates
        phy2torch (torch.Tensor): Transform matrix from physical to normalized coordinates
        device (devicetype): Device where PyTorch tensors are stored
    '''
    def __init__(self, itk_image: sitk.SimpleITK.Image, device: devicetype = 'cuda',
            is_segmentation=False, max_seg_label=None, background_seg_label=0, seg_preprocessor=lambda x: x,
            spacing=None, direction=None, origin=None, center=None) -> None:
        self.itk_image = itk_image
        # check for segmentation parameters
        # if `is_segmentation` is False, then just treat this as a float image
        if not is_segmentation:
            self.array = torch.from_numpy(sitk.GetArrayFromImage(itk_image).astype(float)).to(device).float()
            self.array = self.array[None, None]   # TODO: Change it to support multichannel images, right now just batchify and add a dummy channel to it
            channels = itk_image.GetNumberOfComponentsPerPixel()
            self.channels = channels
            assert channels == 1, "Only single channel images supported"
        else:
            array = torch.from_numpy(sitk.GetArrayFromImage(itk_image).astype(int)).to(device).long()
            # preprocess segmentation if provided by user
            array = seg_preprocessor(array)
            if max_seg_label is not None:
                array[array > max_seg_label] = background_seg_label
            array = integer_to_onehot(array, background_label=background_seg_label, max_label=max_seg_label)[None]  # []
            self.array = array.float()
            self.channels = array.shape[1]
        # initialize matrix for pixel to physical
        dims = itk_image.GetDimension()
        self.dims = dims
        if dims not in [2, 3]:
            raise NotImplementedError("Image class only supports 2D/3D images.")
        
        # custom spacing if not provided use simpleitk values
        spacing = np.array(itk_image.GetSpacing())[None] if spacing is None else np.array(spacing)[None]
        origin = np.array(itk_image.GetOrigin())[None] if origin is None else np.array(origin)[None]
        direction = np.array(itk_image.GetDirection()).reshape(dims, dims) if direction is None else np.array(direction).reshape(dims, dims)
        if center is not None:
            print("Center location provided, recalibrating origin.")
            origin = center - np.matmul(direction, ((np.array(itk_image.GetSize())*spacing/2).squeeze())[:, None]).T

        px2phy = np.eye(dims+1)
        px2phy[:dims, -1] = origin
        px2phy[:dims, :dims] = direction
        px2phy[:dims, :dims] = px2phy[:dims, :dims] * spacing 
        # generate mapping from torch to px
        torch2px = np.eye(dims+1)
        scaleterm = (np.array(itk_image.GetSize())-1)*0.5
        torch2px[:dims, :dims] = np.diag(scaleterm)
        torch2px[:dims, -1] = scaleterm
        # save the mapping from physical to torch and vice versa
        self.torch2phy = torch.from_numpy(np.matmul(px2phy, torch2px)).to(device).float().unsqueeze(0)
        self.phy2torch = torch.inverse(self.torch2phy[0]).float().unsqueeze(0)
        # also save intermediates just in case (as numpy arrays)
        self._torch2px = torch2px
        self._px2phy = px2phy
        self.device = device
        
    @classmethod
    def load_file(cls, image_path:str, *args: Any, **kwargs: Any) -> 'Image':
        '''Load an image from a file.

        Args:
            image_path (str): Path to the image file
            *args: Additional arguments to pass to the Image constructor
            **kwargs: Additional arguments to pass to the Image constructor

        Returns:
            Image: An instance of the Image class
        '''
        itk_image = sitk.ReadImage(image_path)
        return cls(itk_image, *args, **kwargs)
    
    @property
    def shape(self) -> Union[torch.Size, List, Tuple]:
        '''Get the shape of the image.

        Returns:
            torch.Size: Shape of the image
        '''
        return self.array.shape
    
    def delete_array(self):
        '''Delete the PyTorch tensor representation of the image.

        This is a placeholder function to be replaced with batched images.
        '''
        del self.array
    
    def __del__(self):
        '''Delete the SimpleITK image and all intermediate variables.'''
        del self.itk_image
        del self.array
        del self.torch2phy
        del self.phy2torch
        del self._torch2px
        del self._px2phy


class BatchedImages:
    '''A class to handle batches of Image objects efficiently.

    This class provides functionality to work with multiple Image objects as a batch,
    with options for memory optimization and broadcasting. All images in a batch must
    have the same shape.

    Args:
        images (Union[Image, List[Image]]): Single Image object or list of Image objects
        optimize_memory (bool, optional): Flag for memory optimization (reserved for future use)

    Attributes:
        images (List[Image]): List of Image objects in the batch
        n_images (int): Number of images in the batch
        interpolate_mode (str): Interpolation mode ('bilinear' for 2D, 'trilinear' for 3D)
        broadcasted (bool): Whether the batch has been broadcasted
        device (devicetype): Device where the image tensors are stored
        dims (int): Dimensionality of the images (2 or 3)

    Raises:
        ValueError: If batch is empty or images have different shapes
        TypeError: If any element is not an Image object
    '''
    def __init__(self, images: Union[Image, List[Image]], optimize_memory: bool = None) -> None:
        if isinstance(images, Image):
            images = [images]
        self.images = images
        if len(self.images) == 0:
            raise ValueError("BatchedImages must have at least one image")
        for image in self.images:
            if not isinstance(image, Image):
                raise TypeError("All images must be of type Image")
        shapes = [x.array.shape for x in self.images]
        if all([x == shapes[0] for x in shapes]):
            pass
        else:
            raise ValueError("All images must have the same shape")
        self.n_images = len(self.images)
        self.interpolate_mode = 'bilinear' if len(self.images[0].shape) == 4 else 'trilinear'
        self.broadcasted = False

    def __call__(self):
        '''Get the batch of images.
        '''
        if self.broadcasted:
            minusones = [-1] * (len(self.images[0].shape) - 1)
            return self.images[0].array.expand(self.n_images, *minusones)
        else:
            return torch.cat([x.array for x in self.images], dim=0)
    
    def broadcast(self, n):
        '''Broadcast the batch to n channels.

        Args:
            n (int): Number of channels to broadcast to
        
        Raises:
            ValueError: If batch size is not 1
        '''
        if not self.broadcasted and self.n_images != 1:
            raise ValueError("Batch size must be 1 to broadcast")
        self.broadcasted = True
        self.n_images = n
    
    def __del__(self):
        '''Delete all Image objects in the batch.'''
        for image in self.images:
            del image
    
    @property
    def device(self):
        '''Get the device where the image tensors are stored.'''
        return self.images[0].device
    
    @property
    def dims(self):
        '''Get the dimensionality of the images.'''
        return self.images[0].dims
    
    def size(self):
        '''Get the number of images in the batch.'''
        return self.n_images
    
    @property
    def shape(self):
        '''Get the shape of the images.'''
        shape = list(self.images[0].shape)
        shape[0] = self.n_images
        return shape
    
    def get_torch2phy(self):
        return torch.cat([x.torch2phy for x in self.images], dim=0)
    
    def get_phy2torch(self):
        return torch.cat([x.phy2torch for x in self.images], dim=0)


if __name__ == '__main__':
    from glob import glob
    files = sorted(glob("/data/IBSR_braindata/IBSR_01/*nii.gz"))
    image = Image.load_file(files[2])
    print(image.array.shape, image.array.min(), image.array.max())
    # get label
    label = Image.load_file(files[-1], is_segmentation=True)
    print(label.array.shape, label.array.min(), label.array.max())
