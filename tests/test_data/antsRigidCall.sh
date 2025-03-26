antsRegistration --dimensionality 3 --float 1 --output [ANTs/registeredMoment,ANTs/registeredWarpedMoment.nii.gz] --interpolation Linear --winsorize-image-intensities [0.005,0.995] --use-histogram-matching 0  --initial-moving-transform [oasis_157_image.nii.gz,oasis_157_image_rotated.nii.gz,1] --transform Rigid[0.1] \
--metric MeanSquares[oasis_157_image.nii.gz,oasis_157_image_rotated.nii.gz,1,32] \
--convergence [1000x500x250x100,1e-6,10] \
--shrink-factors 8x4x2x1 \
--smoothing-sigmas 3x2x1x0vox 
