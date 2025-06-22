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
from src.datasets.imagenet1k import make_imagenet1k, make_imagenet1k_supervised

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

from src import PKT
from src import which_loss
from src.utils.schedulers import PKTSchedule
from torch.utils.tensorboard import SummaryWriter

import time
import datetime

from sklearn.neighbors import KNeighborsClassifier
# --
log_timings = True
log_freq = 10
# checkpoint_freq = 200
# --

# rng = np.random.Generator(np.random.PCG64())

_GLOBAL_SEED = 2 # 
# seed is logged later on
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# Add this function after the imports but before main()
def evaluate_knn(encoder, train_loader, test_loader, device='cuda'):
    """
    Perform KNN evaluation on model representations
    """
    encoder.eval()
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    
    with torch.no_grad():
        for images, labels in train_loader:
            features = encoder(images.to(device)).mean(dim=1)  # Get embeddings
            train_features.append(features.cpu().numpy())
            train_labels.append(np.array(labels))
            
        for images, labels in test_loader:
            features = encoder(images.to(device)).mean(dim=1)
            test_features.append(features.cpu().numpy())
            test_labels.append(np.array(labels))
    
    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels).astype(int)
    test_features = np.concatenate(test_features)
    test_labels = np.concatenate(test_labels).astype(int)


    # KNN classifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_features, train_labels)

    # accuracy = classifier.score(test_features, test_labels)
    # Calculate train and test accuracy
    train_accuracy = classifier.score(train_features, train_labels)
    test_accuracy = classifier.score(test_features, test_labels)
    
    return train_accuracy, test_accuracy
    
    # return accuracy

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
    evaluate = args['optimization'].get('evaluate', False) # print sim distributions only, do NOT pretrain
    accumulate_grads_every = args['optimization'].get('accumulate_grads_every', 1)
    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    checkpoint_freq = args['logging'].get('checkpoint_freq', 100) # get default frequency, default to 100 otherwise
    logging_frequency = args['logging'].get('logging_frequency', 3) # default to 3
    output_file = args['logging'].get('output_file', tag)
    train_suffix=  args['logging'].get('train_suffix', 'train_images/')
    val_suffix=  args['logging'].get('val_suffix', 'val_images/')
    knn_every = args['logging'].get('knn_every', 50)
    # -- PKT
    """use_pkt_scheduler = args['pkt'].get('use_pkt_scheduler', False)
    start_alpha = args['pkt'].get('start_alpha', 1.)
    warmup_steps_alpha = args['pkt'].get('warmup_steps_alpha', 100)
    ref_alpha = args['pkt'].get('ref_alpha', 1.)
    T_max_alpha = args['pkt'].get('T_max', 200)
    final_alpha = args['pkt'].get('final_alpha', 0.)
    pkt_scale = args['pkt'].get('pkt_scale', 1.0)
    """
    chunks_step = args['pkt'].get('chunks_step', 256)

    # force_cudnn_initialization()
    writer_dest = os.path.join(folder, f'tensorboard-{tag}')
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
    losses_log_file = os.path.join(folder, f'{tag}_all_losses.csv')
    loss_file = os.path.join(folder, f'{tag}_loss_{loss_function}.csv')
    knn_loss_filename = os.path.join(folder, f'{tag}_KNN_{loss_function}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    output_file = os.path.join(folder, output_file)
    logger.addHandler(logging.FileHandler(output_file)) # add auto output ;)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    writer = SummaryWriter(writer_dest)
    csv_logger = CSVLogger(log_file,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%e', 'loss'),
                            ('%.5f', 'mask-A'),
                            ('%.5f', 'mask-B'),
                            ('%d', 'time (ms)'))

    if rank == 0:
        loss_file_logger = CSVLogger(loss_file, 
                                    ('%d', 'epoch'),
                                    ('%e', 'loss'), 
                                    # ('%e', 'loss_L2'),
                                    # ('%e', 'loss_PKT'),
                                    # ('%e', 'cross_mse'),
                                    )
        knn_csv_logger = CSVLogger(knn_loss_filename,
                                    ('%d', 'epoch'),
                                    ('%e', 'knn_accuracy_train'), 
                                    ('%e', 'knn_accuracy_test'), )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

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
            drop_last=True,
            shuffle=True)

    _, train_loader, _ = make_imagenet1k_supervised(
        transform=transform,
        batch_size=batch_size,
        collator=None,  # No mask collator for supervised data
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        shuffle=True,
        train_suffix=train_suffix,
        val_suffix=val_suffix)

    _, test_loader, _ = make_imagenet1k_supervised(
        transform=transform,
        batch_size=batch_size,
        collator=None,  # No mask collator for supervised data
        pin_mem=pin_mem,
        training=False,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        copy_data=copy_data,
        shuffle=False,
        train_suffix=train_suffix,
        val_suffix=val_suffix, )
    ipe = len(unsupervised_loader) # iterations per epoch

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

    """
    # convert T_max_alpha from epoch to actual itr number
    T_max_alpha = ipe*T_max_alpha*ipe_scale
    warmup_steps_alpha = ipe*warmup_steps_alpha*ipe_scale
    pkt_scheduler = PKTSchedule(warmup_steps=warmup_steps_alpha,
                                start_alpha=start_alpha,
                                ref_alpha=ref_alpha,
                                T_max=T_max_alpha,
                                final_alpha=final_alpha)"""

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
            # pkt_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0: # run only once, in main/first process
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0: # or ((epoch + 1) < 200 and (epoch+1) % 10 == 0):
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    start_time = time.perf_counter() # get starting time
    for epoch in range(start_epoch, num_epochs):
        start_time_epoch = time.perf_counter()
        logger.info('Epoch %d' % (epoch + 1))
        encoder.train()
        if logging_frequency > 0:
            log_freq = ipe // logging_frequency # report every X := `logging_frequency` intermediate steps
        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)
        # optimizer.zero_grad() # for safety

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # _new_alpha = pkt_scheduler.step()
                # if rank == 0 and itr == 0 and use_pkt_scheduler:
                #     logger.info('new alpha: %f' % _new_alpha)
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
                    return z

                def loss_fn(z, h, **kwargs):
                    # loss_l2 = F.smooth_l1_loss(z, h) # initial loss
                    # this should be fully functional, as proven by L2
                    final_loss = which_loss.__dict__[loss_function](z,h, **kwargs)

                    # if isinstance(final_loss, tuple): # if more than one loss was returned
                    #     loss_l2 = final_loss[0]
                    #     loss_pkt = final_loss[1]
                    #     mse = final_loss[2]
                    #     loss = AllReduce.apply(loss_l2 + loss_pkt + mse)
                    # else:
                    loss = AllReduce.apply(final_loss)

                    # assert not np.isnan(loss.detach().cpu()), 'NaN loss, abort'
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    # if not use_pkt_scheduler:
                    loss = loss_fn(z, h, chunks_step=chunks_step) # pkt scale default to 1
                    
                    # loss = loss / accumulate_grads_every
                    # else:
                    #     loss = loss_fn(z, h, pkt_scale=pkt_scale, alpha=_new_alpha, chunks_step=chunks_step)
                    # gathered_losses = all_losses(z,h) # this contains all loss functions
                
                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    if (itr+1) % accumulate_grads_every == 0: # or (itr + 1 == len(unsupervised_loader)): # accumulate_grads_every = 1 cancels grad accumulation
                        scaler.step(optimizer)
                        scaler.update()
                        grad_stats = grad_logger(encoder.named_parameters())
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (itr+1) % accumulate_grads_every == 0: # or (itr + 1 == len(unsupervised_loader)):
                        optimizer.step()
                        grad_stats = grad_logger(encoder.named_parameters())
                        optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats, ) #, gathered_losses)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            # neg_var could be None
            loss_meter.update(loss)
            time_meter.update(etime)
            # loss_l2_meter.update(loss_l2)
            # loss_pkt_meter.update(loss_pkt)
            # mse_meter.update(mse)
            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                # custom: all losses report

                if (logging_frequency > 0 and itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %e '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                # 'loss l2: %e '
                                # 'loss pkt: %e '
                                # 'cross mse: %e '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                #    loss_l2_meter.avg,
                                #    loss_pkt_meter.avg,
                                #    mse_meter.avg,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'
        # -- Log loss into appropriate CSV file and to tensorboard
        if rank == 0:
            loss_file_logger.log(epoch+1,
                                 loss_meter.avg,
                                 )
            writer.add_scalar('Loss', loss_meter.avg, epoch+1)
            # writer.add_scalar('Loss L2', loss_l2_meter.avg, epoch+1)
            # writer.add_scalar('Loss PKT', loss_pkt_meter.avg, epoch+1)
            # writer.add_scalar('Cross sim matrix MSE', mse_meter.avg, epoch+1)

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.8e' % loss_meter.avg)
        # logger.info('avg. loss L2: %e avg. loss PKT %e avg. cross sim matrix mse; %e ' % (loss_l2_meter.avg, loss_pkt_meter.avg, mse_meter.avg))
        save_checkpoint(epoch+1)
        if (epoch + 1) % knn_every == 0 and rank == 0:  # Only evaluate KNN on main process
            knn_acc_train, knn_acc_test = evaluate_knn(encoder, train_loader, test_loader, device)
            logger.info(f'\tEpoch {epoch + 1}, KNN accuracy train: {knn_acc_train:.5e}, KNN accuracy train: {knn_acc_test:.5e}')
            knn_csv_logger.log(epoch+1, knn_acc_train, knn_acc_test)
            writer.add_scalar('KNN_Accuracy train', knn_acc_train, epoch + 1)
            writer.add_scalar('KNN_Accuracy test', knn_acc_test, epoch + 1)
        time_epoch = time.perf_counter() - start_time_epoch
        logger.info('time taken for epoch %s' % str(datetime.timedelta(seconds=time_epoch)))
    total_time = time.perf_counter() - start_time
    logger.info('Total pretraining time %s' % str(datetime.timedelta(seconds = total_time)))


if __name__ == "__main__":
    main()
