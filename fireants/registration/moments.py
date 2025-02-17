from fireants.registration.abstract import AbstractRegistration
from typing import List, Optional
import torch
from torch import nn
from fireants.io.image import BatchedImages
from torch.optim import SGD, Adam
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import downsample
from fireants.utils.globals import MIN_IMG_SIZE

class MomentsRegistration(AbstractRegistration):

    def __init__(self, scale: float, 
                fixed_images: BatchedImages, moving_images: BatchedImages,
                # moment matching params
                scaling: bool = False,
                blur: bool = True, 
                moments: int = 1,       # can be 1 or 2
                orientation: str = "rot",           # set to either rot, antirot, or none
                # loss params
                loss_type: str = "cc",
                loss_params: dict = {},
                mi_kernel_type: str = 'b-spline', cc_kernel_type: str = 'rectangular',
                tolerance: float = 1e-6, max_tolerance_iters: int = 10, 
                cc_kernel_size: int = 3,
                custom_loss: nn.Module = None, 
                **kwargs
                ) -> None:
        super().__init__(scales=[scale], iterations=[1], fixed_images=fixed_images, moving_images=moving_images, 
                         loss_type=loss_type, mi_kernel_type=mi_kernel_type, cc_kernel_type=cc_kernel_type, custom_loss=custom_loss, 
                         loss_params=loss_params,
                         cc_kernel_size=cc_kernel_size,
                         reduction='none',  # we want none to evaluate loss for each image for rotation ambiguity
                         tolerance=tolerance, max_tolerance_iters=max_tolerance_iters, **kwargs)
        # initialize transform
        if self.dtype not in [torch.float32, torch.float64]:
            raise ValueError(f"Only float32 and float64 are supported for moments registration, got {self.dtype}")
        if orientation is None:
            orientation = 'both'
        orientation = orientation.lower()
        assert orientation in ['rot', 'antirot', 'both'], "Orientation should be either rot, antirot, or both (None)"
        # set device and dims
        device = fixed_images.device
        self.scaling = scaling
        self.moments = moments
        self.orientation = orientation
        self.dims = self.moving_images.dims
        # introduce some scaling parameter
        self.blur = blur
        self.optimized = False
    
    def compute_second_order_moments(self, arrays, xyz):
        '''
        arrays: [B, HW[D], 1]
        xyz: [B, HW[D], 2/3]
        '''
        arrsum = arrays.flatten(1).sum(1)
        M = torch.zeros(arrays.shape[0], self.dims, self.dims, device=arrays.device)
        # run through all dims (maybe a little slow)
        for i in range(self.dims):
            for j in range(i, self.dims):
                Mij = (arrays[..., 0] * xyz[..., i] * xyz[..., j]).flatten(1).sum(1) / arrsum
                M[:, i, j] = Mij
                if i != j:
                    M[:, j, i] = Mij
        return M.to(self.dtype)
    
    def find_best_detmat_3d(self, U_f, U_m, fixed_arrays, moving_arrays, com_f, com_m, xyz_f, xyz_m):
        ''' 
        Find best determinant matrix for 3D 
        '''
        rot = [-np.eye(3) for _ in range(4)]
        antirot = [np.eye(3) for _ in range(4)]
        for i in range(3):
            rot[i][i, i] = 1
            antirot[i][i, i] = -1
        rot[-1] *= -1
        antirot[-1] *= -1
        if self.orientation == 'rot':
            oris = rot
        elif self.orientation == 'antirot':
            oris = antirot
        else:
            oris = rot + antirot
        oris = np.array(oris)   # [confs, 3, 3]
        oris = torch.tensor(oris, device=fixed_arrays.device).unsqueeze(1).expand(-1, self.opt_size, -1, -1)       # [confs, N, 3, 3]
        oris = oris.to(U_f.dtype)
        moving_p2t = self.moving_images.get_phy2torch().to(U_f.dtype)

        # initialize best idx and best metric for each batch id
        best_idx = torch.zeros(self.opt_size, dtype=torch.long, device=fixed_arrays.device)
        best_metric = torch.zeros(self.opt_size, device=fixed_arrays.device, dtype=fixed_arrays.dtype) + np.inf

        # for each orientation, compute R, t and find best metric
        for ori_id, ori in enumerate(oris):
            # compute R, t
            R = (U_f @ ori @ U_m).to(fixed_arrays.device)   # [N, 3, 3]
            R = R.transpose(-1, -2)
            moved_coords_m = torch.einsum('ntd, n...d->n...t', R, xyz_f) + com_m[:, None]  # [N, S, 3]
            moved_coords_m = torch.einsum('ntd, n...d->n...t', moving_p2t[:, :-1, :-1], moved_coords_m) + moving_p2t[:, :-1, -1].unsqueeze(1)
            # moved_coords_m is now of size [N, S, dims] -> revert it back to [N, H, W, D, dims]
            moved_coords_m = moved_coords_m.view(-1, *fixed_arrays.shape[2:], self.dims)
            # sample moving image?!
            moved_image = F.grid_sample(moving_arrays, moved_coords_m.to(moving_arrays.dtype), mode='bilinear', align_corners=True)
            loss_val = self.loss_fn(moved_image, fixed_arrays).flatten(1).sum(1)
            index = torch.where(loss_val < best_metric)[0]
            best_metric[index] = loss_val[index]
            best_idx[index] = ori_id
        
        # get best orientation  # [N, 3, 3]
        ori = oris[best_idx, 0]
        return ori

    def downsample_images(self, fixed_arrays, moving_arrays):
        ''' Downsample images if scale is more than 1 '''
        if self.scales[0] > 1:
            fixed_size = fixed_arrays.shape[2:]
            moving_size = moving_arrays.shape[2:]
            scale = self.scales[0]
            size_down_f = [max(int(s / scale), MIN_IMG_SIZE) for s in fixed_size]
            size_down_m = [max(int(s / scale), MIN_IMG_SIZE) for s in moving_size]
            # downsample
            if self.blur:
                # blur and downsample for higher scale
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(fixed_size, size_down_f)], device=fixed_arrays.device, dtype=fixed_arrays.dtype)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_arrays = downsample(fixed_arrays, size=size_down_f, mode=self.fixed_images.interpolate_mode, gaussians=gaussians)
                # same for moving images
                sigmas = 0.5 * torch.tensor([sz/szdown for sz, szdown in zip(moving_size, size_down_m)], device=moving_arrays.device, dtype=moving_arrays.dtype)
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                moving_arrays = downsample(moving_arrays, size=size_down_m, mode=self.moving_images.interpolate_mode, gaussians=gaussians)
            else:
                # just downsample
                fixed_arrays = F.interpolate(fixed_arrays, size=size_down_f, mode=self.fixed_images.interpolate_mode, align_corners=True)
                moving_arrays = F.interpolate(moving_arrays, size=size_down_m, mode=self.moving_images.interpolate_mode, align_corners=True)
        return fixed_arrays, moving_arrays

    def get_rigid_transl_init(self):
        ''' Get transform to optimize rigid registration '''
        if not self.optimized:
            raise ValueError("Optimize rigid registration first.")
        return self.tf
    
    def get_rigid_moment_init(self):
        if not self.optimized:
            raise ValueError("Optimize rigid registration first.")
        return self.Rf
    
    def get_affine_init(self):
        if not self.optimized:
            raise ValueError("Optimize rigid registration first.")
        Rf, tf = self.Rf, self.tf # {N, d, d}, {N, d}
        aff = torch.eye(self.dims, self.dims+1, device=self.fixed_images.device,).unsqueeze(0).repeat(Rf.shape[0], 1, 1)
        aff[:, :, :-1] = Rf
        aff[:, :, -1] = tf
        return aff.to(self.dtype)
    
    def optimize_helper(self):
        ''' Optimize rigid registration for 3D images '''
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()

        # downsample if scale is more than 1
        fixed_arrays, moving_arrays = self.downsample_images(fixed_arrays, moving_arrays)

        # get coordinate transforms from pytorch to physical
        fixed_t2p = self.fixed_images.get_torch2phy().to(self.dtype)
        moving_t2p = self.moving_images.get_torch2phy().to(self.dtype)

        # keep track of shapes because we will change the arrays to match coordinate dimensions
        fixed_shape = fixed_arrays.shape
        moving_shape = moving_arrays.shape

        fixed_arrays_imgview = fixed_arrays
        moving_arrays_imgview = moving_arrays

        # take sum along channels and add another channel to keep consistent with coordinates BHWD1
        fixed_arrays = (fixed_arrays.sum(dim=1, keepdim=False).flatten(1, -1))[..., None]
        moving_arrays = (moving_arrays.sum(dim=1, keepdim=False).flatten(1, -1))[..., None]

        # save initial affine transform to initialize grid 
        init_grid_f = torch.eye(self.dims, self.dims+1, device=self.fixed_images.device, dtype=self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, dims, dims+1]
        init_grid_m = torch.eye(self.dims, self.dims+1, device=self.moving_images.device, dtype=self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, dims, dims+1]

        def unsqueeze_last(tensor):
            return tensor.unsqueeze(3) if self.dims == 3 else tensor

        # get initial grids
        # fixed
        xyz_f = F.affine_grid(init_grid_f, fixed_shape, align_corners=True)  # [N, H, W, D, dims]
        xyz_f = torch.einsum('ntd, n...d->n...t', fixed_t2p[:, :-1, :-1], xyz_f)  # [N, H, W, D, dims]
        xyz_f += unsqueeze_last(fixed_t2p[:, :-1, -1].unsqueeze(1).unsqueeze(2))
        # moving
        xyz_m = F.affine_grid(init_grid_m, moving_shape, align_corners=True)  # [N, H, W, D, dims]
        xyz_m = torch.einsum('ntd, n...d->n...t', moving_t2p[:, :-1, :-1], xyz_m)  # [N, H, W, D, dims]
        xyz_m += unsqueeze_last(moving_t2p[:, :-1, -1].unsqueeze(1).unsqueeze(2))  # [N, H, W, D, dims]

        # flatten them along spatial dimensions
        xyz_f = xyz_f.flatten(1, -2)  # [N, H*W*D, dims]
        xyz_m = xyz_m.flatten(1, -2)  # [N, H*W*D, dims]

        # calculate center of mass for fixed and moving images BSC, BS1
        com_f = (fixed_arrays * xyz_f).sum(dim=1) / fixed_arrays.sum(dim=1)
        com_m = (moving_arrays * xyz_m).sum(dim=1) / moving_arrays.sum(dim=1)

        if self.moments == 1:
            # calculate first order moments
            self.tf = -com_f + com_m.to(com_f.device)         # [N, dims]
            self.Rf = torch.eye(self.dims, device=self.fixed_images.device).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, dims, dims]
        elif self.moments == 2:
            # centerize the physical coordinates
            xyz_f = xyz_f - com_f.view(-1, 1, self.dims)
            xyz_m = xyz_m - com_m.view(-1, 1, self.dims)
            # calculate second order moments
            M_f = self.compute_second_order_moments(fixed_arrays, xyz_f)
            M_m = self.compute_second_order_moments(moving_arrays, xyz_m)
            # calculate rotation matrix
            U_f, S_f, Vh_f = torch.linalg.svd(M_f)
            U_m, S_m, Vh_m = torch.linalg.svd(M_m)
            # transpose U_m since the solution is U_f * U_m^T
            U_m = U_m.transpose(-1, -2).to(U_f.device)
            # calculate det
            det = torch.linalg.det(torch.einsum('nij, njk->nik', U_f, U_m))  # N
            detmat = torch.eye(self.dims, device=self.fixed_images.device, dtype=self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)
            detmat[:, -1, -1] = det.to(self.dtype)
            U_f = U_f @ detmat
            if self.scaling:
                raise NotImplementedError("Scaling not implemented for 2nd order moments.")
            # find best rotmat
            if self.dims == 3:
                detmat = self.find_best_detmat_3d(U_f, U_m, fixed_arrays_imgview, moving_arrays_imgview, com_f, com_m, xyz_f, xyz_m)
            elif self.dims == 2:
                detmat = self.find_best_detmat_2d(U_f, U_m, fixed_arrays_imgview, moving_arrays_imgview, com_f, com_m, xyz_f, xyz_m)

            ### note that we calculated x_f = \phi(x_m) but registration is done to warp x_m = \psi(x_f) 
            ### so \psi = \phi^{-1}
            # calculate rotation
            Rf = (U_f @ detmat @ U_m).to(self.fixed_images.device)
            Rf = Rf.transpose(-1, -2)
            self.Rf = Rf
            self.tf = com_m.to(com_f.device) - (com_f[:, None] @ (Rf.transpose(-1, -2))).squeeze(1)
            # self.Rf = (U_f @ detmat @ U_m).to(self.fixed_images.device)
            # self.tf = com_f - com_m.to(com_f.device) @ (self.Rf.transpose(-1, -2))
        else:
            raise NotImplementedError("Only 1st and 2nd order moments supported.")

        self.Rf = self.Rf.to(self.dtype)
        self.tf = self.tf.to(self.dtype)


    def get_warped_coordinates(self, fixed_images: BatchedImages, moving_images: BatchedImages, shape=None):
        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)
        # get affine matrix and append last row
        aff = self.get_affine_init()  # [B, d, d+1]
        row = torch.zeros((self.opt_size, 1, self.dims+1), device=aff.device, dtype=self.dtype)
        row[:, :, -1] = 1
        aff = torch.cat([aff, row], dim=1)  # [B, d+1, d+1]
        # Get shape
        if shape is None:
            shape = fixed_images.shape
        # init grid and apply rigid transformation
        init_grid = torch.eye(self.dims, self.dims+1).to(self.fixed_images.device, self.dtype).unsqueeze(0).repeat(self.opt_size, 1, 1)  # [N, dims, dims+1]
        coords = F.affine_grid(init_grid, shape, align_corners=True)  # [N, H, W, [D], dims+1]
        coords = torch.cat([coords, torch.ones(list(coords.shape[:-1]) + [1], device=coords.device, dtype=self.dtype)], dim=-1)
        coords = torch.einsum('ntd, n...d->n...t', fixed_t2p, coords)  # [N, H, W, [D], dims+1]  
        coords = torch.einsum('ntd, n...d->n...t', aff, coords)  # [N, H, W, [D], dims+1]
        coords = torch.einsum('ntd, n...d->n...t', moving_p2t, coords)  # [N, H, W, [D], dims+1]
        return coords[..., :-1]

    
    def optimize(self, save_transformed=False):
        ''' Given fixed and moving images, optimize rigid registration '''
        if self.optimized:
            print("Already optimized parameters. Use other functions to get transformed values.")
            return
        # optimize
        ret = self.optimize_helper()
        self.optimized = True
        return ret


if __name__ == '__main__':
    from fireants.io.image import Image, BatchedImages
    img_dtype = torch.bfloat16
    img1 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00598/BraTS2021_00598_t1.nii.gz', dtype=img_dtype)
    img2 = Image.load_file('/data/rohitrango/BRATS2021/training/BraTS2021_00599/BraTS2021_00599_t1.nii.gz', dtype=img_dtype)
    fixed = BatchedImages([img1, ])
    moving = BatchedImages([img2,])
    transform = MomentsRegistration(1, fixed, moving, moments=2,)
    transform.optimize()
    rig = transform.get_rigid_transl_init()
    rot = transform.get_rigid_moment_init()
    print(rig.shape, rot.shape)
    print(rig, rot)
    print(torch.linalg.det(rot))