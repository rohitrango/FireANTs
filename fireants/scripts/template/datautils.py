import torch
from torch.utils.data import Dataset, DataLoader
from fireants.registration.distributed import parallel_state
from fireants.io.image import Image, BatchedImages
from fireants.scripts.template.template_helpers import normalize

def fireants_collate_fn(batch):
    '''
    Collate function for fireants
    '''
    # Process each entry in the batch
    processed_batch = {}
    # Get all keys from first item
    keys = batch[0].keys()
    
    for key in keys:
        items = [b[key] for b in batch]
        # Handle different types
        if len(items) > 0:
            if isinstance(items[0], Image):
                # Create BatchedImages for Image objects
                processed_batch[key] = BatchedImages(items)
            elif isinstance(items[0], torch.Tensor):
                # Concatenate tensors along dim 0
                processed_batch[key] = torch.cat([item.unsqueeze(0) for item in items], dim=0)
            else:
                # For all other types, just collect into a list
                processed_batch[key] = items
    return processed_batch

def get_image_dataloader(args):
    '''
    Get a dataloader for the image list file
    '''
    image_list_file = args.image_list_file
    image_prefix = args.image_prefix
    image_suffix = args.image_suffix
    num_subjects = args.num_subjects
    batch_size = args.batch_size
    dataset = MultivariateTemplateBuildingDataset(image_list_file, image_prefix, image_suffix, num_subjects, args)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=fireants_collate_fn)

class MultivariateTemplateBuildingDataset(Dataset):
    '''
    Dataset for multivariate template building
    
    Expects `image_list_file` to contain a comma-separated list of file paths per row.
    - Each non-empty row must have the same number of comma-separated items (k >= 1)
    - Empty lines are ignored
    - Each path is optionally prepended with `image_prefix` and appended with `image_suffix`
    - The final list is cropped to `num_subjects` if provided
    '''
    def __init__(self, image_list_file, image_prefix=None, image_suffix=None, num_subjects=None, args=None):
        self.image_list_file = image_list_file
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.requested_num_subjects = num_subjects
        self.args = args

        self._rows = self._parse_image_list_file(
            image_list_file=image_list_file,
            image_prefix=image_prefix,
            image_suffix=image_suffix,
            num_subjects=num_subjects,
        )
        self.k = len(self._rows[0]) if len(self._rows) > 0 else 0
        self.device = parallel_state.get_device()

    def _apply_prefix_suffix(self, path, prefix, suffix):
        prefixed = f"{prefix}{path}" if prefix is not None else path
        suffixed = f"{prefixed}{suffix}" if suffix is not None else prefixed
        return suffixed

    def _parse_image_list_file(self, image_list_file, image_prefix=None, image_suffix=None, num_subjects=None):
        with open(image_list_file, 'r') as f:
            raw_lines = f.readlines()

        # Strip and ignore empty lines
        lines = [line.strip() for line in raw_lines]
        lines = [line for line in lines if len(line) > 0]

        rows = []
        expected_k = None
        for line_id, line in enumerate(lines):
            # Split by comma, trim whitespace around items
            items = [item.strip() for item in line.split(',')]

            # Skip rows with empty entries
            if any(len(item) == 0 for item in items):
                continue

            # Validate consistent K across rows
            if expected_k is None:
                expected_k = len(items)
                if expected_k < 1:
                    raise ValueError(
                        f"Row {line_id+1} in '{image_list_file}' must contain at least one file path."
                    )
            elif len(items) != expected_k:
                raise ValueError(
                    f"Inconsistent number of items in '{image_list_file}': row 1 has {expected_k}, row {line_id+1} has {len(items)}."
                )

            # Apply prefix/suffix to each item
            processed = [self._apply_prefix_suffix(item, image_prefix, image_suffix) for item in items]
            rows.append(processed)

        # Crop to num_subjects if requested
        if num_subjects is not None:
            rows = rows[:num_subjects]
        
        # split them according to data parallel
        start_rank = parallel_state.get_data_parallel_rank()
        dp_size = parallel_state.get_data_parallel_size()
        self.total_num_rows = len(rows)
        # splice the rows according to data parallel
        rows = rows[start_rank::dp_size]
        return rows

    def __len__(self):
        return len(self._rows)
    
    def __getitem__(self, index):
        # Return the list of file paths for subject `index` (length k)
        imagefiles = self._rows[index]
        identifier = imagefiles[0].split('/')[-1]
        images = [Image.load_file(imagefile, device=self.device) for imagefile in imagefiles]
        if self.args.normalize_images:
            for image in images:
                image.array = normalize(image.array.data)

        images[0].concatenate(*images[1:], optimize_memory=True)
        return {
            'image': images[0],
            'identifier': identifier,
        }
