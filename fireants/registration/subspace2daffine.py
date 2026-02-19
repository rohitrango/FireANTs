#!/usr/bin/env python

# Copyright (c) 2026 Rohit Jena. All rights reserved.
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


import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


from fireants.io.image import BatchedImages, FakeBatchedImages
from fireants.registration.abstract import AbstractRegistration


class Subspace2DAffineRegistration(AbstractRegistration):
    """2D single-channel affine registration using subspace (SVD) matching of contours.

    This registration class estimates an affine transform in physical space by:
        - extracting object silhouettes via OpenCV contours,
        - converting contour coordinates from voxel to physical space, and
        - matching the resulting point sets via an SVD-based subspace method.

    The resulting affine matrices are stored in physical coordinates in the
    convention y = A x + t, mapping fixed-image physical coordinates x to
    moving-image physical coordinates y, consistent with `AffineRegistration`.
    """

    def __init__(
        self,
        fixed_images: BatchedImages,
        moving_images: BatchedImages,
        scale: Optional[int] = 1,
        loss_type: str = "cc",
        loss_params: Optional[dict] = None,
        mi_kernel_type: str = "gaussian",
        cc_kernel_type: str = "rectangular",
        cc_kernel_size: int = 3,
        tolerance: float = 1e-6,
        max_tolerance_iters: int = 10,
        custom_loss: Optional[nn.Module] = None,
        orientation: str = "rot",
        **kwargs,
    ) -> None:
        if loss_params is None:
            loss_params = {}

        # We call the parent with a single dummy scale/iteration, mirroring MomentsRegistration.
        super().__init__(
            scales=[scale],
            iterations=[1],
            fixed_images=fixed_images,
            moving_images=moving_images,
            loss_type=loss_type,
            mi_kernel_type=mi_kernel_type,
            cc_kernel_type=cc_kernel_type,
            custom_loss=custom_loss,
            loss_params=loss_params,
            cc_kernel_size=cc_kernel_size,
            reduction="none",
            tolerance=tolerance,
            max_tolerance_iters=max_tolerance_iters,
            **kwargs,
        )


        assert orientation in ["rot", "antirot", "both"], "Orientation must be 'rot', 'antirot', or 'both'"
        self.orientation = orientation

        if self.dims != 2:
            raise ValueError(
                f"Subspace2DAffineRegistration supports only 2D images, got dims={self.dims}"
            )

        # Enforce single channel (plus optional mask channel in masked mode).
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_channels = fixed_arrays.shape[1]
        moving_channels = moving_arrays.shape[1]

        if fixed_channels != 1 or moving_channels != 1:
            raise ValueError(
                "Subspace2DAffineRegistration supports only single-channel 2D images"
            )

        device = self.device
        dtype = self.dtype

        # Store affine in physical space: [N, 2, 3], y = A x + t
        self.affine = torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0).repeat(
            self.opt_size, 1, 1
        )
        self.optimized = False

    def get_affine_matrix(self, homogenous: bool = True) -> torch.Tensor:
        """Return the current affine matrices in physical space.

        Args:
            homogenous: If True, returns [N, 3, 3] homogenous matrices.
                        If False, returns [N, 2, 3] matrices.
        """
        if not homogenous:
            return self.affine.contiguous()

        row = torch.zeros(
            (self.opt_size, 1, 3), device=self.affine.device, dtype=self.affine.dtype
        )
        row[:, :, -1] = 1.0
        return torch.cat([self.affine, row], dim=1).contiguous()

    def _get_numpy_images_for_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed and moving images as numpy arrays in voxel space.

        Shapes: [N, H, W]
        """
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()

        fixed_np = fixed_arrays[:, 0].detach().cpu().numpy()
        moving_np = moving_arrays[:, 0].detach().cpu().numpy()
        return fixed_np, moving_np

    def _get_silhouette(self, binary_mask: torch.Tensor) -> torch.Tensor:
        """Extract ordered contour points (x, y) from a binary mask as a torch tensor.

        Args:
            binary_mask: [H, W] boolean or uint8 tensor in voxel space.

        Returns:
            torch.Tensor: [2, N] tensor of (x, y) points on the contour, on the
            same device and dtype as the registration object.
        """
        import cv2  # Imported lazily to avoid hard dependency at module import time

        if binary_mask.dtype != torch.uint8:
            binary_mask = binary_mask.to(torch.uint8)

        mask_np = binary_mask.detach().cpu().numpy()

        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            # Return empty tensor if no contour is found.
            raise ValueError("No contours found in the binary mask")

        largest_contour = max(contours, key=cv2.contourArea)
        silhouette_points = largest_contour.reshape(-1, 2).astype(np.float32)  # [N, 2]
        pts = torch.from_numpy(silhouette_points.T)  # [2, N]
        return pts.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _normalize_shape(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize a 2xN point cloud and return (centroid, A, Vt)."""
        # points: [2, N]
        centroid = points.mean(dim=1, keepdim=True)  # [2, 1]
        points_centered = points - centroid  # [2, N]

        # SVD over 2 x N
        U, S, Vt = torch.linalg.svd(points_centered, full_matrices=False)
        # Normalize as in the draft script
        norm = torch.linalg.vector_norm(Vt, dim=0).mean()
        eps = torch.finfo(points.dtype).eps
        norm = torch.clamp(norm, min=eps)
        Vt_norm = Vt / norm
        S_scaled = S * norm
        A = U @ torch.diag(S_scaled)  # [2, 2]
        return centroid[:, 0], A, Vt_norm

    @staticmethod
    def _robust_chamfer_dist(
        points1: torch.Tensor, points2: torch.Tensor, percentile: Optional[int] = 90
    ) -> float:
        """Robust Chamfer distance between two 2xN point sets."""
        # points: [2, N]
        # dist_mat: [N1, N2]
        dist_mat = (
            (points1**2).sum(0).unsqueeze(1)
            + (points2**2).sum(0).unsqueeze(0)
            - 2.0 * (points1.T @ points2)
        )
        dist_mat = torch.sqrt(torch.clamp(dist_mat, min=0.0) + 1e-9)
        if percentile is None:
            d1 = dist_mat.min(dim=0).values.mean()
            d2 = dist_mat.min(dim=1).values.mean()
            return float(d1 + d2)

        dist1 = dist_mat.min(dim=0).values
        k1 = max(1, int(percentile / 100.0 * dist1.shape[0]))
        dist1_sorted, _ = torch.sort(dist1)
        dist1_val = dist1_sorted[:k1].mean()

        dist2 = dist_mat.min(dim=1).values
        k2 = max(1, int(percentile / 100.0 * dist2.shape[0]))
        dist2_sorted, _ = torch.sort(dist2)
        dist2_val = dist2_sorted[:k2].mean()
        return float(dist1_val + dist2_val)

    @classmethod
    def _find_best_rotation(
        cls,
        points1: torch.Tensor,
        points2: torch.Tensor,
        degrees: int = 1,
        orientation: str = "rot",
    ) -> torch.Tensor:
        """Brute-force search over rotations (and optional flips) to align two shapes.

        Orientation semantics mirror `MomentsRegistration`:
            - orientation='rot'      → pure rotations (det = +1)
            - orientation='antirot'  → flipped rotations (det = -1)
            - orientation='both'     → consider both
        """
        device = points1.device
        dtype = points1.dtype
        best_R = None
        best_dist = float("inf")

        # Map legacy flip / fliponly flags to orientation if not explicitly provided
        assert orientation in ["rot", "antirot", "both"], "Orientation must be 'rot', 'antirot', or 'both'"
        orientation = orientation.lower()

        def _rot(angle_rad: float) -> torch.Tensor:
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            return torch.tensor([[c, -s], [s, c]], device=device, dtype=dtype)

        # Define orientation sets similar to MomentsRegistration (2D case).
        rot = [torch.eye(2, device=device, dtype=dtype)]
        antirot = [torch.tensor([[-1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)]

        if orientation == "rot":
            oris = rot
        elif orientation == "antirot":
            oris = antirot
        elif orientation == "both":
            oris = rot + antirot
        else:
            raise ValueError(f"Unknown orientation '{orientation}', expected 'rot', 'antirot', or 'both'.")

        for ori in oris:
            for deg in tqdm(range(0, 360, degrees), desc="Rotation Search"):
                R = _rot(math.radians(deg)) @ ori
                p2 = R @ points2
                dist = cls._robust_chamfer_dist(points1, p2)
                if dist < best_dist:
                    best_dist = dist
                    best_R = R

        if best_R is None:
            best_R = torch.eye(2, device=device, dtype=dtype)
        return best_R

    @staticmethod
    def _get_composite_affine_transform(
        fixed_params: tuple[torch.Tensor, torch.Tensor],
        moving_params: tuple[torch.Tensor, torch.Tensor],
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose affine A, t from fixed and moving shape parameters.

        All inputs are in physical coordinates.
        """
        centroid_fixed, A_fixed = fixed_params
        centroid_moving, A_moving = moving_params

        A = A_moving @ R @ torch.linalg.inv(A_fixed)
        t = centroid_moving - A @ centroid_fixed
        return A, t

    def _optimize_helper(self) -> None:
        """Compute affine transforms for each batch element in a single shot."""
        if self.optimized:
            return

        # Get images and geometry on the current device.
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_imgs = fixed_arrays[:, 0]
        moving_imgs = moving_arrays[:, 0]  # [N, H, W]

        # Use px2phy to convert voxel coordinates to physical coordinates.
        # px2phy: [N, 3, 3] for 2D images.
        fixed_px2phy = self.fixed_images.px2phy.to(self.dtype)
        moving_px2phy = self.moving_images.px2phy.to(self.dtype)

        affines = []
        for b in range(self.opt_size):
            fixed_img = fixed_imgs[b]
            moving_img = moving_imgs[b]

            # Simple thresholding to get binary masks; caller can pre-mask if desired.
            fixed_mask = fixed_img > 0
            moving_mask = moving_img > 0

            fixed_pts_px = self._get_silhouette(fixed_mask)
            moving_pts_px = self._get_silhouette(moving_mask)

            if fixed_pts_px.numel() == 0 or moving_pts_px.numel() == 0:
                # Fallback to identity if silhouettes cannot be computed.
                raise ValueError("No contours found in the binary mask")

            # Convert from pixel to physical coordinates.
            # px2phy has shape [N, 3, 3]; we use slice for batch b.
            px2phy_fixed = fixed_px2phy[b]
            px2phy_moving = moving_px2phy[b]

            def to_phys(px_pts: torch.Tensor, px2phy_mat: torch.Tensor) -> torch.Tensor:
                # px_pts: [2, N]
                ones = torch.ones(
                    1, px_pts.shape[1], device=px_pts.device, dtype=px_pts.dtype
                )
                homog = torch.cat([px_pts, ones], dim=0)  # [3, N]
                phy = px2phy_mat @ homog  # [3, N]
                return phy[:2]

            fixed_pts_phy = to_phys(fixed_pts_px, px2phy_fixed)
            moving_pts_phy = to_phys(moving_pts_px, px2phy_moving)

            cf, Af, fixed_Vt = self._normalize_shape(fixed_pts_phy)
            cm, Am, moving_Vt = self._normalize_shape(moving_pts_phy)

            # Find best rotation (allow flipping only, similar to draft).
            R = self._find_best_rotation(
                fixed_Vt, moving_Vt, degrees=1, orientation=self.orientation
            )

            A, t = self._get_composite_affine_transform((cf, Af), (cm, Am), R)

            aff = torch.cat([A, t.view(2, 1)], dim=1)  # [2, 3]
            affines.append(aff)

        self.affine = torch.stack(affines, dim=0).to(device=self.device, dtype=self.dtype)
        self.optimized = True

    def get_warp_parameters(
        self,
        fixed_images: Union[BatchedImages, FakeBatchedImages],
        moving_images: Union[BatchedImages, FakeBatchedImages],
        shape=None,
    ):
        """Get warp parameters in the format expected by `fireants_interpolator`.

        Returns an affine defined in torch-normalized coordinates via:
            affine = moving_phy2torch @ A_phys @ fixed_torch2phy
        """
        if not self.optimized:
            raise ValueError("Call optimize() before requesting warp parameters.")

        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)

        aff = self.get_affine_matrix(homogenous=True)  # [N, 3, 3]
        affine = ((moving_p2t @ aff @ fixed_t2p)[:, :-1, :]).contiguous().to(self.dtype)

        if shape is None:
            shape = fixed_images.shape

        return {
            "affine": affine,
            "out_shape": shape,
        }

    def get_inverse_warp_parameters(
        self,
        fixed_images: Union[BatchedImages, FakeBatchedImages],
        moving_images: Union[BatchedImages, FakeBatchedImages],
        shape=None,
    ):
        if not self.optimized:
            raise ValueError("Call optimize() before requesting warp parameters.")

        fixed_t2p = fixed_images.get_torch2phy().to(self.dtype)
        moving_p2t = moving_images.get_phy2torch().to(self.dtype)

        aff = self.get_affine_matrix(homogenous=True)  # [N, 3, 3]
        affine = ((moving_p2t @ aff @ fixed_t2p))
        affine_inv = torch.linalg.inv(affine)
        affine_inv = affine_inv[:, :-1, :].contiguous().to(self.dtype)

        if shape is None:
            shape = fixed_images.shape

        return {
            "affine": affine_inv,
            "out_shape": shape,
        }

    def optimize(self):
        """Compute affine transforms once for all batch elements."""
        if self.optimized:
            return
        self._optimize_helper()

