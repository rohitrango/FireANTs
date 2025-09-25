import pytest
import torch
import SimpleITK as sitk
import numpy as np
from fireants.io.image import Image, BatchedImages, FakeBatchedImages, concat
from fireants.registration.distributed import parallel_state
import os

## needs to be run with `torchrun --nproc_per_node=1 test_image_io.py`

@pytest.fixture
def sample_itk_image_2d():
    """Create a sample 2D SimpleITK image for testing"""
    size = [100, 100]
    spacing = [1.0, 1.0]
    origin = [0.0, 0.0]
    direction = [1.0, 0.0, 0.0, 1.0]  # 2x2 identity matrix flattened
    
    # Create a simple gradient image
    img_array = np.zeros(size[::-1], dtype=np.float32)  # SimpleITK uses [y,x] ordering
    for i in range(size[1]):
        for j in range(size[0]):
            img_array[i,j] = i + j
            
    itk_image = sitk.GetImageFromArray(img_array)
    itk_image.SetSpacing(spacing)
    itk_image.SetOrigin(origin)
    itk_image.SetDirection(direction)
    return itk_image

@pytest.fixture
def sample_itk_image_3d():
    """Create a sample 3D SimpleITK image for testing"""
    size = [50, 50, 50]
    spacing = [1.0, 1.0, 1.0]
    origin = [0.0, 0.0, 0.0]
    direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 3x3 identity matrix flattened
    
    # Create a simple gradient image
    img_array = np.zeros(size[::-1], dtype=np.float32)  # SimpleITK uses [z,y,x] ordering
    for i in range(size[2]):
        for j in range(size[1]):
            for k in range(size[0]):
                img_array[i,j,k] = i + j + k
                
    itk_image = sitk.GetImageFromArray(img_array)
    itk_image.SetSpacing(spacing)
    itk_image.SetOrigin(origin)
    itk_image.SetDirection(direction)
    return itk_image

@pytest.fixture
def sample_seg_image_3d():
    """Create a sample 3D segmentation image for testing"""
    size = [50, 50, 50]
    img_array = np.zeros(size[::-1], dtype=np.int32)
    # Create some segmentation regions
    img_array[10:20, 10:20, 10:20] = 1
    img_array[30:40, 30:40, 30:40] = 2
    
    itk_image = sitk.GetImageFromArray(img_array)
    return itk_image

class TestImage:
    def test_init_2d(self, sample_itk_image_2d):
        """Test basic Image initialization with 2D image"""
        img = Image(sample_itk_image_2d, device='cpu')
        assert img.dims == 2
        assert img.channels == 1
        assert len(img.shape) == 4  # [B, C, H, W]
        assert img.array.shape[0] == 1  # batch size
        assert img.array.shape[1] == 1  # channels
        assert torch.is_tensor(img.array)
        assert img.device == 'cpu'
        
    def test_init_3d(self, sample_itk_image_3d):
        """Test basic Image initialization with 3D image"""
        img = Image(sample_itk_image_3d, device='cpu')
        assert img.dims == 3
        assert img.channels == 1
        assert len(img.shape) == 5  # [B, C, H, W, D]
        assert img.array.shape[0] == 1  # batch size
        assert img.array.shape[1] == 1  # channels
        assert torch.is_tensor(img.array)
        
    def test_segmentation(self, sample_seg_image_3d):
        """Test segmentation image handling"""
        img = Image(sample_seg_image_3d, device='cpu', is_segmentation=True)
        assert img.channels == 2
        assert img.array.shape[1] == 2
        # Check that values are binary (0 or 1)
        assert torch.all(torch.logical_or(img.array == 0, img.array == 1))
        # Check that regions with label 1 or 2 are marked as foreground (1)
        orig_array = sitk.GetArrayFromImage(sample_seg_image_3d)
        binary_array = (orig_array > 0).astype(np.float32)
        assert np.array_equal(img.array.squeeze().cpu().max(dim=0).values.numpy(), binary_array)
        
    def test_coordinate_transforms(self, sample_itk_image_3d):
        """Test coordinate transformation matrices"""
        img = Image(sample_itk_image_3d, device='cpu')
        assert img.torch2phy.shape == (1, 4, 4)  # homogeneous coordinates
        assert img.phy2torch.shape == (1, 4, 4)
        # Check that they are inverses
        identity = torch.eye(4).unsqueeze(0)
        assert torch.allclose(torch.matmul(img.torch2phy, img.phy2torch), identity)
        
    def test_array_management(self, sample_itk_image_2d):
        """Test array presence and deletion"""
        img = Image(sample_itk_image_2d, device='cpu')
        assert img.is_array_present
        img.delete_array()
        assert not img.is_array_present
        
    def test_device_movement(self, sample_itk_image_2d):
        """Test moving image between devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        img = Image(sample_itk_image_2d, device='cpu')
        img.to('cuda')
        assert img.device == 'cuda'
        assert img.array.device.type == 'cuda'
        img.to('cpu')
        assert img.device == 'cpu'
        assert img.array.device.type == 'cpu'

class TestBatchedImages:
    def test_init_single(self, sample_itk_image_2d):
        """Test BatchedImages initialization with single image"""
        img = Image(sample_itk_image_2d, device='cpu')
        batch = BatchedImages(img)
        assert batch.n_images == 1
        assert batch.dims == 2
        assert not batch.broadcasted
        assert batch.interpolate_mode == 'bilinear'
        
    def test_init_multiple(self, sample_itk_image_2d):
        """Test BatchedImages initialization with multiple images"""
        imgs = [Image(sample_itk_image_2d, device='cpu') for _ in range(3)]
        batch = BatchedImages(imgs)
        assert batch.n_images == 3
        assert batch.dims == 2
        assert not batch.broadcasted
        assert batch.batch_tensor.shape[0] == 3
        
    def test_broadcasting(self, sample_itk_image_2d):
        """Test broadcasting functionality"""
        img = Image(sample_itk_image_2d, device='cpu')
        batch = BatchedImages(img)
        batch.broadcast(5)
        assert batch.n_images == 5
        assert batch.broadcasted
        output = batch()
        assert output.shape[0] == 5
        
    def test_sharding(self, sample_itk_image_3d):
        """Test sharding functionality"""
        # Initialize parallel state for testing
        parallel_state.initialize_parallel_state(data_parallel_size=1, override_torchrun_check=True)
        
        img = Image(sample_itk_image_3d, device='cpu')
        batch = BatchedImages(img)
        assert not batch.is_sharded
        
        # Test sharding along a dimension
        batch._shard_dim(0)  # shard along H dimension, rank 0
        assert batch.is_sharded
        assert hasattr(batch, '_shard_start')
        assert hasattr(batch, '_shard_end')
        assert hasattr(batch, '_dim_to_shard')
        
        # Cleanup
        parallel_state.cleanup_parallel_state()

class TestFakeBatchedImages:
    def test_init(self, sample_itk_image_2d):
        """Test FakeBatchedImages initialization"""
        img = Image(sample_itk_image_2d, device='cpu')
        batch = BatchedImages(img)
        tensor = torch.randn_like(batch())
        
        fake_batch = FakeBatchedImages(tensor, batch)
        assert fake_batch.is_sharded == batch.is_sharded  # Should be the same as batch sharded status
        assert torch.equal(fake_batch(), tensor)
        assert fake_batch.dims == batch.dims
        assert fake_batch.shape == tensor.shape
        
    def test_metadata_inheritance(self, sample_itk_image_2d):
        """Test that FakeBatchedImages inherits metadata correctly"""
        img = Image(sample_itk_image_2d, device='cpu')
        batch = BatchedImages(img)
        tensor = torch.randn_like(batch())
        
        fake_batch = FakeBatchedImages(tensor, batch)
        assert torch.equal(fake_batch.get_torch2phy(), batch.torch2phy)
        assert torch.equal(fake_batch.get_phy2torch(), batch.phy2torch)

def test_concat_images(sample_itk_image_2d):
    """Test image concatenation functionality"""
    # Create multiple images
    imgs = [Image(sample_itk_image_2d, device='cpu') for _ in range(3)]
    
    # Test concatenation with optimize_memory=True
    result = concat(*imgs, optimize_memory=True)
    assert result.channels == 3
    assert result.shape[0] == 1

    assert not any(img.is_array_present for img in imgs[1:])  # Arrays should be deleted
    
    # Test concatenation with optimize_memory=False
    imgs = [Image(sample_itk_image_2d, device='cpu') for _ in range(3)]
    result = concat(*imgs, optimize_memory=False)
    assert result.channels == 3
    assert all(img.is_array_present for img in imgs)  # Arrays should still be present
