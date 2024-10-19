''' author: rohitrango

helper functions for the template building script
'''

import time
import os

class Timer:
    def __init__(self, description: str, logger):
        self.description = description
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        if self.logger is not None:
            self.logger.info(f'"{self.description}" took {elapsed_time:.2f} seconds.')

def check_args(args, logger):
    ''' 
    Parameters:
    args: argparse.Namespace
        arguments passed to the script
    logger: logging.Logger
    '''
    # save tmpdir
    if args.tmpdir is None:
        args.tmpdir = os.environ.get('TMPDIR', '/tmp')
    
    # check for last registration 
    if args.do_deform:
        args.last_reg = 'deform'
    elif args.do_affine:
        args.last_reg = 'affine'
    elif args.do_rigid:
        args.last_reg = 'rigid'
    else:
        logger.error('No registration type specified')
        return False

    return True

def get_image_files(args):
    ''' get image files and process them '''
    with open(args.image_file, 'r') as f:
        image_files = f.readlines()
    image_files = list(filter(lambda x: len(x) > 0, [f.strip() for f in image_files]))
    if args.image_prefix is not None:
        image_files = [args.image_prefix + f for f in image_files]
    if args.image_suffix is not None:
        image_files = [f + args.image_suffix for f in image_files]
    return image_files