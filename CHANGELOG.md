# Changelog

All notable changes to this project will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## 2024-03-12

Added initial multimodal support.

```python
t1 = Image.load_file("/data/t1.nii.gz")
t2 = Image.load_file("/data/t2.nii.gz")
t1ce = Image.load_file("/data/t1ce.nii.gz")
flair = Image.load_file("/data/flair.nii.gz")
t1.concatenate(t2, t1ce, flair)		# in-place concatenation

# do stuff with t1
```

## 2024-12-27

Added first iteration of documentation.

## [0.0.1] - 2024-12-06

Added freeform deformations and utility to save to scipy format.
