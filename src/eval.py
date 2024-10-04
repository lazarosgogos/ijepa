# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

from src import PKT
from src import which_loss
from torch.utils.tensorboard import SummaryWriter

import time
import datetime
# --
log_timings = True
log_freq = 10
# checkpoint_freq = 200
# --

# rng = np.random.Generator(np.random.PCG64()) 

_GLOBAL_SEED = 0 
# seed is logged later on
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()



def force_cudnn_initialization():
    return
    s = 32
    print('Forced cudnn initialization')
    dev = torch.device('cuda:0')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    # run_svd = args['meta'].get('svd', False)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    loss_function = args['optimization'].get('loss_function', 'L2') # get the loss function, use L2 if no loss fn definition was found in the config file
    # evaluate = args['optimization'].get('evaluate', False) # print sim distributions only, do NOT pretrain

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    checkpoint_freq = args['logging'].get('checkpoint_freq', 100) # get default frequency, default to 100 otherwise
    logging_frequency = args['logging'].get('logging_frequency', 3) # default to 3
    output_file = args['logging'].get('output_file', tag)
    plot_matrices = args['logging'].get('plot_matrices', True)
    tensorboard_dir = folder # args['logging'].get('tensorboard_dir', 'runs/')
    use_tensorboard = args['logging'].get('use_tensorboard', False)

    # force_cudnn_initialization()
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)
    logger.info(f'train.py: {_GLOBAL_SEED=}') # log seed
    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    output_file = os.path.join(folder, output_file)
    # tensorboard_dir = os.path.join(folder, tensorboard_dir)
    logger.addHandler(logging.FileHandler(output_file)) # add auto output ;)


    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path



    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        use_tensorboard=use_tensorboard,
        tensorboard_dir=tensorboard_dir
        )
    import re
    match_ = re.search(r'ep(\d+)', r_file) # extract the number based on the checkpoint
    pretrain_epoch = match_.group(1)
    target_encoder = copy.deepcopy(encoder)
    if use_tensorboard:
        encoder.init_summary_writer('context_encoder', pretrain_epoch)
        target_encoder.init_summary_writer('target_encoder', pretrain_epoch)
        predictor.init_summary_writer('predictor', pretrain_epoch)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()


    # -- TRAINING LOOP
    start_time = time.perf_counter() # get starting time
    # for epoch in range(start_epoch, num_epochs):
    start_time_epoch = time.perf_counter()
    logger.info('Starting')


    
    all_model_sims, all_target_sims = [], []
    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
        logger.info('Iteration: %d' % itr)
        if itr == 1: break
        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)
            masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
            return (imgs, masks_1, masks_2)
        imgs, masks_enc, masks_pred = load_imgs()

        def train_step():
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            # --

            def forward_target():
                with torch.no_grad():
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                    B = len(h)
                    # -- create targets (masked regions of h)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                    return h

            def forward_context():
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                logger.critical('z[:100] values: %s' % (str(z[:100])))
                return z

            def loss_fn(z, h):
                # this should be fully functional, as proven by L2 
                final_loss = which_loss.__dict__[loss_function](z,h)
                # loss_l2 = F.smooth_l1_loss(z, h) # initial loss
                loss = AllReduce.apply(final_loss)
                return loss
                

            # Step 1. Forward
            with torch.no_grad():
                h = forward_target()
                z = forward_context()
                z = z.view(64, 4, *z.size()[1:])
                h = h.view(64, 4, *h.size()[1:])
                logger.info(h.size())
                
                z = z[0]
                h = h[0] # get ALL blocks from 0th (first) image, theoretically
                logger.info(h.size())
                z = z.view(-1, 768)
                h = h.view(-1, 768)
                
                model_sim, target_sim = PKT.get_similarity_matrices(z, h)

            return model_sim, target_sim
        (model_sim, target_sim), etime = gpu_timer(train_step)

        all_model_sims. append(model_sim.detach().cpu().numpy())
        all_target_sims.append(target_sim.detach().cpu().numpy())

    # logger.info('All model similarities: %s', str(all_model_sims[:5000]))
    # logger.info('All target similarities: %s',str(all_target_sims[:5000]))

    # after all iterations
    # save_checkpoint(epoch+1)


    # -- Visualize weights using Summary Writer - old method with no filtering# 
    # ep = '-ep100'
    import re
    match_ = re.search(r'ep(\d+)', r_file) # extract the number based on the checkpoint
    ep = match_.group(1)
    writer = SummaryWriter(f'runs/l2-{ep}')
    # logger.critical(str(len(all_model_sims)))
    all_params = []
    for idx, (name, param) in enumerate(encoder.named_parameters()):
        # logger.info('epoch: %s, name: %s, param: %s ' 
        #             % (ep, name, param))
        # logger.info('extending params epoch: %s, name: %s' % (ep, name))
        all_params.extend(param.view(-1).detach().cpu().numpy())
        writer.add_histogram(name, param, 
                             global_step=idx, bins=1000)

    outfile_params = os.path.join(folder, f'params-ep{ep}.png')
    ub = max(all_params)
    lb = min(all_params)
    plt.figure(figsize=(10,10),dpi=300)
    plt.yscale('log')
    plt.hist(all_params, bins=1000, range=(lb, ub))
    # plt.hist(all_params,)
    plt.title('Params')
    plt.ylabel('param count')
    plt.xlabel('param value')
    # plt.savefig(outfile_params)
    plt.close()
    del all_params
    writer.close()

    if plot_matrices:
        ri = torch.randint(0, len(all_model_sims), (1,)) # pick a random image from the batch
        lb = min(all_model_sims[ri].min(), all_target_sims[ri].min()) # lower bound
        ub = max(all_model_sims[ri].max(), all_target_sims[ri].max()) # upper bound
        lb = 0.
        ub = 1.
        fig = plt.figure(figsize=(20,10),dpi=300)
        data = (all_model_sims[ri], all_target_sims[ri])
        titles = ('Predictions', 'Targets')
        for idx, datum in enumerate(data):
            plt.subplot(1,2,idx+1)
            img = plt.imshow(datum, interpolation='nearest', vmin=lb, vmax=ub)
            plt.colorbar(img, fraction=0.046, pad=0.04)
            plt.title(titles[idx])
        plt.show()
        plt.suptitle('Evaluation of %s' % (r_file))
        # outfile = os.path.join(folder, f'sims-PKT-ep{ep}.png')
        outfile = os.path.join(folder, f'matrices-inbefore-1ims-1-16-test-PKT-maxvar100-ep{ep}.png')
        plt.savefig(outfile)
        fig.clear()
        plt.close()
    """
    fig = plt.figure()
    plt.hist(all_model_sims, bins=100, range=(0. , 1.), fc=(0, 0, 1, 0.5), label='Predicted sims')
    # , range=(0,1)
    outfile = os.path.join(folder, f'model-sims-ep{ep}.png')
    # plt.savefig(outfile)

    plt.hist(all_target_sims, bins=100, range=(0. , 1.), fc=(1, 0, 0, 0.5), label='Target sims')
    plt.legend()
    """
    
    """
    plt.xlabel('Values distribution')
    plt.ylabel('Count')
    """
    
    time_epoch = time.perf_counter() - start_time_epoch
    logger.info('time taken for epoch %s' % str(datetime.timedelta(seconds=time_epoch)))


    total_time = time.perf_counter() - start_time
    logger.info('Total pretraining time %s' % str(datetime.timedelta(seconds = total_time)))
    

if __name__ == "__main__":
    main()
