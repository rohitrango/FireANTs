#!/bin/bash
export PATH=~/tk/lddmm/xc64rel/:$PATH

# Affine registration
greedy -d 3 -i atlas_2mm_1000_3.nii.gz atlas_2mm_1001_3.nii.gz -a -o affine_fx_1000_mv_1001.mat -m NCC 2x2x2 -n 100x40

# Deformable registration
greedy -d 3 -i atlas_2mm_1000_3.nii.gz atlas_2mm_1001_3.nii.gz -it affine_fx_1000_mv_1001.mat -m NCC 2x2x2 -n 100x40 -wp 0 -sv -o warp_fx_1000_mv_1001.nii.gz -oroot  svf_fx_1000_mv_1001.nii.gz

# Reslice using affine only
greedy -d 3 -rf atlas_2mm_1000_3.nii.gz -rm atlas_2mm_1001_3.nii.gz reslice_affine_fx_1000_mv_1001.nii.gz -r affine_fx_1000_mv_1001.mat

# Reslice using affine and deformable
greedy -d 3 -rf atlas_2mm_1000_3.nii.gz -rm atlas_2mm_1001_3.nii.gz reslice_deform_fx_1000_mv_1001.nii.gz -r warp_fx_1000_mv_1001.nii.gz affine_fx_1000_mv_1001.mat
