# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
os.environ["WORLD_SIZE"] = "1"


import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main
from src.eval import main as evall
# from src.test import main as test_app_main
import time
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

parser.add_argument( # 
    '--eval', type=int, default=0, 
    help='Whether to eval the model without any training',
    required=False
)


def process_main(rank, fname, world_size, devices, test=0):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    if test == 1:
        logger.critical('EVALUATING')
        import glob
        # this r_file has the prefix!
        r_file = params['meta']['read_checkpoint']
        log_dir = params['logging'].get('folder', None)
        r_file = os.path.join(log_dir, r_file)
        tarfiles = glob.glob(r_file + '-ep*.pth.tar')
        
        import re # import regex
        def relevant(v):

            m = re.search(r'-ep(\d+)', v) # find the epoch number
            m = int(m.group(1))
            rel = [10, 20, 100, 200, 300, 400, 500] 
            # rel = [100]
            if m in rel: # if the epoch number is in the relevant ones
                return True
            else: 
                return False
        
        # tarfiles = list(filter(relevant, tarfiles)) # this should only grab the relevant file
        logger.info('tarfiles: ' + str(tarfiles))
        
        # tarfiles has a  list of names of all tarballs with this desired prefix
        import copy
        epoch = 0
        for tarfile in sorted(tarfiles):
            temp_params = copy.deepcopy(params)
            logger.info('working on file %s out of %s...' % (str(tarfile), str(tarfiles)))
            temp_params['meta']['read_checkpoint'] = os.path.basename(tarfile)
            evall(args=temp_params)
    else:
        logger.critical('PRETRAINING')
        app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    try:
        test = args.eval # try to read test cl argument
    except:
        test = 0 # set it to false by default
    
    # if num_gpus == 1:
    #     # no DDP
    #     process_main(0, args.fname, 1, 1, test)

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, test)
            # args=(rank, args.fname, num_gpus, args.devices)
        ).start()
    
