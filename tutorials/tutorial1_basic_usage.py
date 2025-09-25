#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
from time import time
from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.greedy import GreedyRegistration
from fireants.interpolator import fireants_interpolator
from fireants.utils.imageutils import jacobian

def main():
    # Load the images
    print("Loading images...")
    image1 = Image.load_file("atlas_2mm_1000_3.nii.gz")
    image2 = Image.load_file("atlas_2mm_1001_3.nii.gz")
    
    # Batchify them (we only have a single image per batch, but we can pass multiple images)
    batch1 = BatchedImages([image1])
    batch2 = BatchedImages([image2])
    
    # Check device name
    print(f"Using device: {batch1().device}")
    
    # Print coordinate transformation matrices
    print("\nCoordinate transformation matrices:")
    print("phy2torch:")
    print(image1.phy2torch)
    print("\ntorch2phy:")
    print(image1.torch2phy)
    print("\nphy2torch @ torch2phy:")
    print(image1.phy2torch @ image1.torch2phy)
    
    # Perform affine registration
    print("\nPerforming affine registration...")
    scales = [4, 2, 1]  # scales at which to perform registration
    iterations = [200, 100, 50]
    optim = 'Adam'
    lr = 3e-3
    
    # Create affine registration object
    affine = AffineRegistration(scales, iterations, batch1, batch2, 
                              optimizer=optim, optimizer_lr=lr,
                              cc_kernel_size=5)
    
    # Run registration
    start = time()
    affine.optimize()
    moved = affine.evaluate(batch1, batch2)
    end = time()
    print(f"Affine registration runtime: {end - start:.2f} seconds")
    
    # Perform deformable registration
    print("\nPerforming deformable registration...")
    reg = GreedyRegistration(scales=[4, 2, 1], iterations=[200, 100, 25],
                           fixed_images=batch1, moving_images=batch2,
                           cc_kernel_size=5, deformation_type='compositive',
                           smooth_grad_sigma=1,
                           optimizer='adam', optimizer_lr=0.5,
                           init_affine=affine.get_affine_matrix().detach())
    
    # Run deformable registration
    start = time()
    reg.optimize()
    end = time()
    print(f"Deformable registration runtime: {end - start:.2f} seconds")
    
    # Get moved image
    moved = reg.evaluate(batch1, batch2)
    
    # Save warp as ANTs image
    # Compare FireANTs and ANTs results
    print("\nChecking diffeomorphism...")
    warp = reg.get_warped_coordinates(batch1, batch2)
    jac = jacobian(warp).permute(0, 2, 3, 4, 1, 5)[:, 1:-1, 1:-1, 1:-1, :]
    det = torch.linalg.det(jac).reshape(-1).data.cpu().numpy()
    print(f"Percentage of non-positive determinants: {(det<=0).mean()*100:.2f}%")
    

if __name__ == "__main__":
    main() 
