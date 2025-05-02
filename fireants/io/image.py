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
from fireants.utils.util import check_and_raise_cond, augment_filenames
import logging
from copy import deepcopy
import os
from fireants.utils.globals import PERMITTED_ANTS_WARP_EXT
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def divide_size_into_chunks(size: int, world_size: int) -> list:
    '''
    Divide a size into as-equal-as-possible chunks based on world size for distributed
    
    Args:
        size (int): The total size to be divided
        world_size (int): Number of processes to divide the size across
        
    Returns:
        list: List of chunk sizes that sum up to the original size
    '''
    base_size = size // world_size
    remainder = size % world_size
    chunks = [base_size] * world_size
    
    # Distribute remainder across first 'remainder' processes
    for i in range(remainder):
        chunks[i] += 1
    return chunks

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
        orientation (str, optional): Reorient the image to this orientation. Defaults to None.
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
    def __init__(self, itk_image: sitk.SimpleITK.Image, 
                 device: devicetype = 'cuda', 
                 dtype: torch.dtype = None,
                 is_segmentation=False, max_seg_label=None, 
                 background_seg_label=0, seg_preprocessor=lambda x: x,
                 orientation: str = None,
                spacing=None, direction=None, origin=None, center=None) -> None:

        if orientation is not None:
            itk_image = sitk.DICOMOrient(itk_image, orientation)

        self.itk_image = itk_image
        if dtype is None:
            dtype = torch.float32
        # check for segmentation parameters
        # if `is_segmentation` is False, then just treat this as an image with given dtype
        if not is_segmentation:
            self.array = torch.from_numpy(sitk.GetArrayFromImage(itk_image).astype(float)).to(device, dtype)
            # self.array = self.array[None, None]   # TODO: Change it to support multichannel images, right now just batchify and add a dummy channel to it
            channels = itk_image.GetNumberOfComponentsPerPixel()
            self.channels = channels
            # assert channels == 1, "Only single channel images supported"
            if channels > 1:
                logger.warning("Image has multiple channels, make sure its not a spatial dimension")
                # permute the channel dimension to the front
                ndim = self.array.ndim
                self.array = self.array.permute([ndim-1] + list(range(ndim-1))).contiguous() # permute to [C, H, W, D] from [H, W, D, C]
            else:
                self.array.unsqueeze_(0)
            # add batch dimension
            self.array.unsqueeze_(0)
        else:
            array = torch.from_numpy(sitk.GetArrayFromImage(itk_image).astype(int)).to(device).long()
            # preprocess segmentation if provided by user
            array = seg_preprocessor(array)
            if max_seg_label is not None:
                array[array > max_seg_label] = background_seg_label
            array = integer_to_onehot(array, background_label=background_seg_label, max_label=max_seg_label, dtype=dtype)[None]  # [1, C, H, W, D]
            self.array = array
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
        self.torch2phy = torch.from_numpy(np.matmul(px2phy, torch2px)).to(device).float().unsqueeze_(0)
        self.phy2torch = torch.inverse(self.torch2phy[0]).float().unsqueeze_(0)
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
    
    @property
    def is_array_present(self) -> bool:
        '''Check if the PyTorch tensor representation of the image is present.
        This is needed because the BatchedImages may delete the array of the image.

        Returns:
            bool: True if the PyTorch tensor representation of the image is present, False otherwise
        '''
        return hasattr(self, 'array')

    def delete_array(self):
        '''Delete the PyTorch tensor representation of the image.

        This is a placeholder function to be replaced with batched images.
        '''
        if self.is_array_present:
            del self.array
    
    def concatenate(self, *others, optimize_memory: bool = True):
        ''' 
        others is a list of Images or list of lists (in which case the user accidentally passed a list of images) 

            optimize_memory: if True, delete the arrays of the other images after concatenation

        Example:
            t1 = Image.load_file(t1_path)
            t2 = Image.load_file(t2_path)
            flair = Image.load_file(flair_path)
            t1.concatenate(t2, flair, optimize_memory=True)   # deletes the arrays of t2 and flair after concatenation
        '''
        check_and_raise_cond(self.is_array_present, "Image must have a PyTorch tensor representation to concatenate", ValueError)
        if isinstance(others[0], list) and len(others) == 1:
            others = others[0]
        else:
            check_and_raise_cond(all([isinstance(other, Image) for other in others]), "All elements of others must be of type Image", TypeError)
        # check if all images have the same shape
        shapes = [x.array.shape[2:] for x in others]
        base_shape = self.array.shape[2:]
        check_and_raise_cond(all([x == base_shape for x in shapes]), "All images must have the same shape", ValueError)
        check_and_raise_cond(all([x.is_array_present for x in others]), "All images must have a PyTorch tensor representation", ValueError)
        check_and_raise_cond(all([self.array.device == other.array.device for other in others]), "Images must be on the same device", ValueError)
        check_and_raise_cond(all([torch.allclose(self.phy2torch, other.phy2torch) for other in others]), "Images reside in different physical spaces, using the first image's physical space", logger.warning)

        self.array = torch.cat([self.array] + [other.array for other in others], dim=1)
        if optimize_memory:
            logger.debug("Deleting the arrays of the other images after concatenation")
            for other in others:
                other.delete_array()

    def __del__(self):
        '''Delete the SimpleITK image and all intermediate variables.'''
        del self.itk_image
        if self.is_array_present:
            del self.array
        del self.torch2phy
        del self.phy2torch
        del self._torch2px
        del self._px2phy

def concat(*images, optimize_memory: bool = True):
    ''' Creates a copy of the images and concatenates them along the channel dimension
    '''
    check_and_raise_cond(len(images) > 0, "At least one image must be provided", ValueError)
    check_and_raise_cond(all([isinstance(image, Image) for image in images]), "All images must be of type Image", TypeError)
    img0 = images[0] if optimize_memory else deepcopy(images[0])
    img0.concatenate(*images[1:], optimize_memory=optimize_memory)
    return img0

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
        # check if all images have a PyTorch tensor representation
        check_and_raise_cond(all([image.is_array_present for image in self.images]), "All images must have a PyTorch tensor representation", ValueError)
        # Check if all images are of type Image
        check_and_raise_cond(all([isinstance(image, Image) for image in self.images]), "All images must be of type Image", TypeError)
        # check if all images have the same shape
        shapes = [x.array.shape[2:] for x in self.images]
        check_and_raise_cond(all([x == shapes[0] for x in shapes]), "All images must have the same shape", ValueError)
        # set the number of images
        self.n_images = len(self.images)
        self.interpolate_mode = 'bilinear' if len(self.images[0].shape) == 4 else 'trilinear'
        self.broadcasted = False
        # create a batch tensor
        self.batch_tensor = torch.cat([x.array for x in self.images], dim=0) if self.n_images > 1 else self.images[0].array
        # delete the arrays of the images if multiple images
        if optimize_memory and len(self.images) > 1:
            logger.info("Deleting the arrays of the images")
            for image in self.images:
                image.delete_array()
        # create metadata
        self.torch2phy = torch.cat([x.torch2phy for x in self.images], dim=0)
        self.phy2torch = torch.cat([x.phy2torch for x in self.images], dim=0)
        # sharding info 
        self.is_sharded = False
    
    def set_device(self, device_name):
        # set the device for the batch tensor
        self.batch_tensor = self.batch_tensor.to(device_name)
        self.torch2phy = self.torch2phy.to(device_name)
        self.phy2torch = self.phy2torch.to(device_name)
    
    def _shard_dim(self, dim_to_shard, rank, world_size):
        size = self.batch_tensor.shape[dim_to_shard+2]
        chunk_sizes = divide_size_into_chunks(size, world_size)
        # check 
        start = sum(chunk_sizes[:rank])
        end = sum(chunk_sizes[:rank+1])
        # store the full tensor into batch_tensor_full, and sharded version in 
        self.batch_tensor_full = self.batch_tensor
        self.batch_tensor = self.batch_tensor_full.narrow_copy(dim_to_shard+2, start, end-start)
        print(f"Sharded batch tensor shape: {self.batch_tensor.shape} and rank: {rank}/{world_size}")
        self._shard_start = start
        self._shard_end = end
        self._shard_dim = dim_to_shard
        self.is_sharded = True
    
    def _save_shards(self, dim_to_shard, world_size):
        size = self.batch_tensor.shape[dim_to_shard+2]
        chunk_size = (size + world_size - 1) // world_size
        chunks = []
        for _ in range(world_size):
            start = _ * chunk_size
            end = min(start + chunk_size, size)
            chunks.append((start, end))
        self._sharded_chunks = chunks


    def __call__(self):
        '''Get the batch of images.
        '''
        if self.broadcasted:
            minusones = [-1] * (len(self.images[0].shape) - 1)
            return self.batch_tensor.expand(self.n_images, *minusones)
        else:
            return self.batch_tensor
    
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
        del self.batch_tensor
        del self.torch2phy
        del self.phy2torch
    
    @property
    def device(self):
        '''Get the device where the image tensors are stored.'''
        return self.batch_tensor.device
    
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
        # shape = list(self.images[0].shape)
        shape = list(self.batch_tensor.shape)
        shape[0] = self.n_images
        return shape
    
    def get_torch2phy(self):
        return self.torch2phy
    
    def get_phy2torch(self):
        return self.phy2torch

class FakeBatchedImages:
    '''     
    A class to handle fake batches of images.
    This is used to handle the case where the user passes a tensor to the registration class
    instead of a BatchedImages object.

    We will use the metadata of the BatchedImages object to create a FakeBatchedImages object.
    with the content of the tensor.
    '''
    def __init__(self, tensor: torch.Tensor, batched_images: BatchedImages, ignore_size_match: bool = False) -> None:
        batched_size = list(deepcopy(batched_images().shape))
        tensor_size = list(deepcopy(tensor.shape))
        # ignore channel dimension differences
        batched_size[1] = 1
        tensor_size[1] = 1
        if not ignore_size_match:
            check_and_raise_cond(tuple(batched_size) == tuple(tensor_size), "Tensor size must match the size of the batched images", ValueError)
        self.tensor = tensor
        self.batched_images = batched_images
        self.is_sharded = True   # assume that it inherits a sharded image
    
    def __call__(self):
        return self.tensor
    
    def get_torch2phy(self):
        return self.batched_images.torch2phy
    
    def get_phy2torch(self):
        return self.batched_images.phy2torch
    
    @property
    def device(self):
        return self.tensor.device
    
    @property
    def dims(self):
        return self.tensor.ndim - 2
    
    @property
    def shape(self):
        return self.tensor.shape
    
    def write_image(self, filenames: Union[str, List[str]], permitted_ext: List[str] = PERMITTED_ANTS_WARP_EXT):
        """
        Save tensor elements to disk as SimpleITK images.
        
        For each image in the batch:
        - If multi-channel, the channel dimension is permuted to the end
        - If single-channel, the channel dimension is squeezed
        - Metadata is copied from the corresponding BatchedImages itk_image
        
        Args:
            filenames (str or List[str]): A single filename or a list of filenames.
                - If one filename is provided for multiple images, they will be saved as 
                  filename_img0.ext, filename_img1.ext, etc.
                - If the number of filenames equals the number of images, they are mapped one-to-one.
                - Otherwise, an error is raised.
        
        Raises:
            ValueError: If the number of filenames doesn't match the number of images and is not 1.
        """
        batch_size = self.tensor.shape[0]
        
        # Convert single filename to list
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # Check if number of filenames matches number of images
        check_and_raise_cond(len(filenames)==1 or len(filenames)==batch_size, "Number of filenames must match the number of images or be 1", ValueError)
        filenames = augment_filenames(filenames, batch_size, permitted_ext)
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get the corresponding tensor
            img_tensor = self.tensor[i]
            
            # Check if multi-channel (channel dimension is at index 0 after batch dimension)
            channels = img_tensor.shape[0]
            isVector = channels > 1

            # If multi-channel, permute the channel to the end
            if channels > 1:
                # For 2D: [C, H, W] -> [H, W, C]
                # For 3D: [C, H, W, D] -> [H, W, D, C]
                dims = len(img_tensor.shape)
                perm = list(range(1, dims)) + [0]
                img_tensor = img_tensor.permute(*perm)
            else:
                # If single channel, squeeze the channel dimension
                img_tensor = img_tensor.squeeze(0)
            
            # Convert tensor to numpy array
            np_array = img_tensor.detach().cpu().numpy()
            
            # Create SimpleITK image
            itk_image = sitk.GetImageFromArray(np_array, isVector=isVector)
            
            # Get metadata from corresponding BatchedImages object
            if hasattr(self.batched_images, 'images') and i < len(self.batched_images.images):
                src_itk = self.batched_images.images[i].itk_image
                itk_image.SetSpacing(src_itk.GetSpacing())
                itk_image.SetDirection(src_itk.GetDirection())
                itk_image.SetOrigin(src_itk.GetOrigin())
            else:
                raise ValueError("No corresponding BatchedImages object found for image {}".format(i))
            
            save_filename = filenames[i]
            # Save the image
            sitk.WriteImage(itk_image, save_filename)
            logger.info(f"Saved image to {save_filename}")

if __name__ == '__main__':
    from fireants.utils.util import get_tensor_memory_details
    from glob import glob
    import os
    # files = sorted(glob(f"{os.environ['DATAPATH_R']}/IBSR_braindata/IBSR_01/*nii.gz"))
    file = f"{os.environ['DATA_PATH2']}/fMOST/subject/15257_red_mm_IRA.nii.gz"
    # torch.cuda.memory._record_memory_history()

    image = Image.load_file(file)
    details = get_tensor_memory_details()
    for tensor, size, _, _ in details:
        print(tensor.shape, size)
    
    # torch.cuda.memory._dump_snapshot("image.pkl")
    # print(image.array.shape, image.array.min(), image.array.max())
    # # get label
    # label = Image.load_file(files[-1], is_segmentation=True)
    # print(label.array.shape, label.array.min(), label.array.max())

    # Check concatenation
    mem_start = torch.cuda.memory_allocated()
    t1 = Image.load_file(f"{os.environ['DATAPATH_R']}/BRATS2021/training/BraTS2021_00624/BraTS2021_00624_t1.nii.gz")
    t2 = Image.load_file(f"{os.environ['DATAPATH_R']}/BRATS2021/training/BraTS2021_00624/BraTS2021_00624_t2.nii.gz")
    t1ce = Image.load_file(f"{os.environ['DATAPATH_R']}/BRATS2021/training/BraTS2021_00624/BraTS2021_00624_t1ce.nii.gz")
    flair = Image.load_file(f"{os.environ['DATAPATH_R']}/BRATS2021/training/BraTS2021_00624/BraTS2021_00624_flair.nii.gz")
    t1.concatenate(t2, t1ce, flair)
    print(t1.array.shape)
    print(t2.is_array_present, t1ce.is_array_present, flair.is_array_present)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_end = torch.cuda.memory_allocated()
    print(f"Memory allocated: {(mem_end - mem_start)/1024**2:.4f} MB")
    details = get_tensor_memory_details()
    for tensor, size, _, _ in details:
        print(tensor.shape, size)

