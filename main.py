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
        evall(args=params)
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
    
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, test)
            # args=(rank, args.fname, num_gpus, args.devices)
        ).start()
    
