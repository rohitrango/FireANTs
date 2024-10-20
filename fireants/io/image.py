import torch
import SimpleITK as sitk
import numpy as np
from typing import Any, Union, List
from time import time
from fireants.types import devicetype
from fireants.utils.imageutils import integer_to_onehot

class Image:
    '''
    TODO: Documentation here
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
    def load_file(cls, image_path:str, *args, **kwargs) -> 'Image':
        itk_image = sitk.ReadImage(image_path)
        return cls(itk_image, *args, **kwargs)
    
    @property
    def shape(self):
        return self.array.shape
    
    def delete_array(self):
        # safely delete the array, change this later with batchedimages
        del self.array
    
    def __del__(self):
        del self.itk_image
        del self.array
        del self.torch2phy
        del self.phy2torch
        del self._torch2px
        del self._px2phy


class BatchedImages:
    '''
    Class for batched images
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
        # get batch of images, this consumes more memory
        if self.broadcasted:
            minusones = [-1] * (len(self.images[0].shape) - 1)
            return self.images[0].array.expand(self.n_images, *minusones)
        else:
            return torch.cat([x.array for x in self.images], dim=0)
    
    def broadcast(self, n):
        # broadcast the batch to n channels, only works if the batch size is 1 to begin with
        if not self.broadcasted and self.n_images != 1:
            raise ValueError("Batch size must be 1 to broadcast")
        self.broadcasted = True
        self.n_images = n
    
    def __del__(self):
        for image in self.images:
            del image
    
    @property
    def device(self):
        return self.images[0].device
    
    @property
    def dims(self):
        return self.images[0].dims
    
    def size(self):
        return self.n_images
    
    @property
    def shape(self):
        shape = list(self.images[0].shape)
        shape[0] = self.n_images
        return shape
    
    def get_torch2phy(self):
        return torch.cat([x.torch2phy for x in self.images], dim=0)
    
    def get_phy2torch(self):
        return torch.cat([x.phy2torch for x in self.images], dim=0)


if __name__ == '__main__':
    # image = Image.load_file('/data/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz')
    # print(image.torch2phy)
    # image2 = Image.load_file('/data/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t1.nii.gz')
    # batch = BatchedImages([image, image2])
    # print(batch().shape)
    # print(batch.get_torch2phy().shape)
    from glob import glob
    files = sorted(glob("/data/IBSR_braindata/IBSR_01/*nii.gz"))
    image = Image.load_file(files[2])
    print(image.array.shape, image.array.min(), image.array.max())
    # get label
    label = Image.load_file(files[-1], is_segmentation=True)
    print(label.array.shape, label.array.min(), label.array.max())
