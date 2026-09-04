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
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
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
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import load_checkpoint, init_model, init_opt
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


def representation_metrics(X, eps=1e-12):
    """
    X: [N, D]
    """

    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)

    N, D = X.shape

    C = (X.T @ X) / max(N - 1, 1)

    eigvals = torch.linalg.eigvalsh(C)
    eigvals = torch.clamp(eigvals, min=0)

    probs = eigvals / (eigvals.sum() + eps)

    entropy = -(probs * torch.log(probs + eps)).sum()

    effective_rank = torch.exp(entropy)

    max_rank = min(N - 1, D)

    normalized_entropy = entropy / torch.log(
        torch.tensor(float(max_rank), device=X.device)
    )

    normalized_effective_rank = effective_rank / max_rank

    return {
        "vne": entropy.item(),
        "vne_norm": normalized_entropy.item(),
        "effective_rank": effective_rank.item(),
        "effective_rank_norm": normalized_effective_rank.item(),
        "eigenvalues": eigvals.detach().cpu(),
    }


def compute_tail_metrics(losses, top_fracs=(0.01, 0.05, 0.10)):
    """
    losses: 1D tensor of per-patch losses, shape [N]

    Returns:
        dict containing total loss concentration statistics.
    """

    losses = losses.float().detach().cpu()
    losses = losses[torch.isfinite(losses)]

    sorted_losses, _ = torch.sort(losses, descending=True)

    total_loss = sorted_losses.sum().item()
    mean_loss = sorted_losses.mean().item()
    median_loss = sorted_losses.median().item()
    max_loss = sorted_losses.max().item()

    results = {
        "num_patches": int(sorted_losses.numel()),
        "total_loss": total_loss,
        "mean_loss": mean_loss,
        "median_loss": median_loss,
        "max_loss": max_loss,
    }

    for frac in top_fracs:
        k = max(1, int(frac * sorted_losses.numel()))
        top_loss = sorted_losses[:k].sum().item()
        results[f"top_{int(frac * 100)}pct_loss_share"] = top_loss / total_loss

    return results


def _as_float_list(values):
    """Parse corruption ratios from config values such as [0, .05] or ['0%', '5%']."""
    parsed = []
    for value in values:
        if isinstance(value, str):
            value = value.strip()
            if value.endswith("%"):
                parsed.append(float(value[:-1]) / 100.0)
            else:
                parsed.append(float(value))
        else:
            parsed.append(float(value))
    return parsed


def objective_loss_per_patch(z, h, loss_name):
    """
    Return one scalar objective value for already-flattened predicted and target
    representations. This lets the same sensitivity probe evaluate L2 and the
    alternative objectives exposed through src.which_loss.
    """
    if loss_name == "L2":
        return ((z - h) ** 2).mean(dim=-1)
    else:
        return PKT.cosine_similarity_loss(z, h)

    # if hasattr(which_loss, loss_name):
    #     loss = which_loss.__dict__[loss_name](z, h)
    # else:
    #     # Fallback used by the existing evaluation code for non-L2 objectives.
    #     loss = PKT.cosine_similarity_loss(z, h)

    # if torch.is_tensor(loss) and loss.numel() > 1:
    #     loss = loss.mean()
    # return loss


def per_patch_objective_loss(z, h, loss_name):
    """Return one loss value per patch/token for tail analysis."""
    if loss_name == "L2":
        return ((z - h) ** 2).mean(dim=-1)

    loss = PKT.cosine_similarity_loss(z, h)
    if torch.is_tensor(loss) and loss.ndim == 0:
        # Some relational objectives only expose a batch scalar. Replicate it so
        # downstream tail-analysis code remains well-defined, though L2 is the
        # meaningful per-patch diagnostic.
        loss = loss.expand(z.size(0))
    return loss.reshape(-1)


def corrupt_target_representations(h, corruption_ratio, batch_size, generator=None):
    """
    Replace a random fraction of target patch representations with target
    representations from different images in the same batch.

    Returns:
        h_corrupted: same shape as h
        corruption_mask_flat: bool tensor of shape [num_patches]
            True only for patches that were corrupted.
    """

    r = float(corruption_ratio)

    out = h.clone()
    original_shape = out.shape

    if batch_size < 2:
        raise ValueError("Target corruption requires batch_size >= 2.")

    if out.dim() == 3:
        first_dim, tokens, dim = out.shape

        if first_dim % batch_size != 0:
            raise ValueError(
                f"Cannot infer image groups for target shape {tuple(original_shape)} "
                f"and batch_size={batch_size}."
            )

        groups = first_dim // batch_size
        view = out.view(batch_size, groups, tokens, dim)

    elif out.dim() == 2:
        rows, dim = out.shape

        if rows % batch_size != 0:
            raise ValueError(
                f"Cannot infer image groups for target shape {tuple(original_shape)} "
                f"and batch_size={batch_size}."
            )

        groups = rows // batch_size
        tokens = 1
        view = out.view(batch_size, groups, tokens, dim)

    else:
        raise ValueError(
            f"Unsupported target shape for corruption: {tuple(original_shape)}"
        )

    device = out.device
    num_positions = batch_size * groups * tokens

    corruption_mask_flat = torch.zeros(
        num_positions,
        dtype=torch.bool,
        device=device,
    )

    if r <= 0.0:
        return out.view(original_shape), corruption_mask_flat

    num_corrupt = int(round(r * num_positions))
    num_corrupt = max(1, min(num_corrupt, num_positions))

    flat_indices = torch.randperm(
        num_positions,
        device=device,
        generator=generator,
    )[:num_corrupt]

    corruption_mask_flat[flat_indices] = True

    b = flat_indices // (groups * tokens)
    rem = flat_indices % (groups * tokens)
    g = rem // tokens
    t = rem % tokens

    offsets = torch.randint(
        low=1,
        high=batch_size,
        size=(num_corrupt,),
        device=device,
        generator=generator,
    )

    src_b = (b + offsets) % batch_size

    view[b, g, t] = view[src_b, g, t]

    return out.view(original_shape), corruption_mask_flat


def summarize_per_patch_sensitivity(
    clean_patch_loss, corrupted_patch_loss, corruption_mask
):
    """
    Summarize per-patch corruption sensitivity.

    clean_patch_loss:     [num_patches]
    corrupted_patch_loss: [num_patches]
    corruption_mask:      [num_patches], bool
    """
    eps = 1e-12

    clean_patch_loss = clean_patch_loss.detach()
    corrupted_patch_loss = corrupted_patch_loss.detach()
    corruption_mask = corruption_mask.detach().bool()

    abs_delta = corrupted_patch_loss - clean_patch_loss
    rel_delta = abs_delta / clean_patch_loss.clamp_min(eps)

    result = {
        "all_clean_mean": clean_patch_loss.mean().item(),
        "all_corrupted_mean": corrupted_patch_loss.mean().item(),
        "all_abs_delta_mean": abs_delta.mean().item(),
        "all_rel_delta_mean": rel_delta.mean().item(),
        "all_rel_delta_median": rel_delta.median().item(),
        "all_rel_delta_p95": torch.quantile(rel_delta.float(), 0.95).item(),
        "all_rel_delta_p99": torch.quantile(rel_delta.float(), 0.99).item(),
        "num_patches": int(clean_patch_loss.numel()),
        "num_corrupted_patches": int(corruption_mask.sum().item()),
    }

    if corruption_mask.any():
        corrupted_only_abs = abs_delta[corruption_mask]
        corrupted_only_rel = rel_delta[corruption_mask]

        result.update(
            {
                "corrupted_only_abs_delta_mean": corrupted_only_abs.mean().item(),
                "corrupted_only_abs_delta_median": corrupted_only_abs.median().item(),
                "corrupted_only_abs_delta_p95": torch.quantile(
                    corrupted_only_abs.float(), 0.95
                ).item(),
                "corrupted_only_abs_delta_p99": torch.quantile(
                    corrupted_only_abs.float(), 0.99
                ).item(),
                "corrupted_only_rel_delta_mean": corrupted_only_rel.mean().item(),
                "corrupted_only_rel_delta_median": corrupted_only_rel.median().item(),
                "corrupted_only_rel_delta_p95": torch.quantile(
                    corrupted_only_rel.float(), 0.95
                ).item(),
                "corrupted_only_rel_delta_p99": torch.quantile(
                    corrupted_only_rel.float(), 0.99
                ).item(),
            }
        )
    else:
        result.update(
            {
                "corrupted_only_abs_delta_mean": 0.0,
                "corrupted_only_abs_delta_median": 0.0,
                "corrupted_only_abs_delta_p95": 0.0,
                "corrupted_only_abs_delta_p99": 0.0,
                "corrupted_only_rel_delta_mean": 0.0,
                "corrupted_only_rel_delta_median": 0.0,
                "corrupted_only_rel_delta_p95": 0.0,
                "corrupted_only_rel_delta_p99": 0.0,
            }
        )

    return result


def compute_sensitivity_row(
    z, h, corruption_ratio, batch_size, objective_names, generator=None
):
    """Compute clean/corrupted/relative-loss-increase values for one corruption ratio."""
    h_corrupt = corrupt_target_representations(
        h=h,
        corruption_ratio=corruption_ratio,
        batch_size=batch_size,
        generator=generator,
    )

    row = {"corruption_ratio": float(corruption_ratio)}
    for name in objective_names:
        clean = objective_loss_scalar(z, h, name).detach()
        corrupted = objective_loss_scalar(z, h_corrupt, name).detach()
        denom = clean.clamp_min(1e-12)
        rel_inc = (corrupted - clean) / denom
        row[f"{name}_clean"] = clean.item()
        row[f"{name}_corrupted"] = corrupted.item()
        row[f"{name}_relative_increase"] = rel_inc.item()
    return row


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args["meta"]["use_bfloat16"]
    model_name = args["meta"]["model_name"]
    load_model = args["meta"]["load_checkpoint"] or resume_preempt
    r_file = args["meta"]["read_checkpoint"]
    copy_data = args["meta"]["copy_data"]
    pred_depth = args["meta"]["pred_depth"]
    pred_emb_dim = args["meta"]["pred_emb_dim"]
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    # run_svd = args['meta'].get('svd', False)

    # -- DATA
    use_gaussian_blur = args["data"]["use_gaussian_blur"]
    use_horizontal_flip = args["data"]["use_horizontal_flip"]
    use_color_distortion = args["data"]["use_color_distortion"]
    color_jitter = args["data"]["color_jitter_strength"]
    # --
    batch_size = args["data"]["batch_size"]
    pin_mem = args["data"]["pin_mem"]
    num_workers = args["data"]["num_workers"]
    root_path = args["data"]["root_path"]
    image_folder = args["data"]["image_folder"]
    crop_size = args["data"]["crop_size"]
    crop_scale = args["data"]["crop_scale"]
    # --

    # -- MASK
    allow_overlap = args["mask"][
        "allow_overlap"
    ]  # whether to allow overlap b/w context and target blocks
    patch_size = args["mask"]["patch_size"]  # patch-size for model training
    num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
    min_keep = args["mask"]["min_keep"]  # min number of patches in context block
    enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
    num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
    pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
    aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args["optimization"]["ema"]
    ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
    wd = float(args["optimization"]["weight_decay"])
    final_wd = float(args["optimization"]["final_weight_decay"])
    num_epochs = args["optimization"]["epochs"]
    warmup = args["optimization"]["warmup"]
    start_lr = args["optimization"]["start_lr"]
    lr = args["optimization"]["lr"]
    final_lr = args["optimization"]["final_lr"]
    loss_function = args["optimization"].get(
        "loss_function", "L2"
    )  # get the loss function, use L2 if no loss fn definition was found in the config file

    # -- SENSITIVITY EVALUATION
    # Implements the PDF experiment: frozen model, corrupt target representations,
    # and measure relative objective loss increase.
    sensitivity_cfg = args.get("sensitivity", {})
    run_sensitivity = sensitivity_cfg.get("enabled", True)
    corruption_ratios = _as_float_list(
        sensitivity_cfg.get("corruption_ratios", [0.0, 0.05, 0.10, 0.20, 0.30])
    )
    # Compare baseline L2 with the configured objective by default. Override with:
    # sensitivity:
    #   objectives: ['L2', 'PKT']
    sensitivity_objectives = sensitivity_cfg.get("objectives", ["L2", loss_function])
    sensitivity_objectives = list(dict.fromkeys(sensitivity_objectives))
    # evaluate = args['optimization'].get('evaluate', False) # print sim distributions only, do NOT pretrain

    # -- LOGGING
    folder = args["logging"]["folder"]
    tag = args["logging"]["write_tag"]
    checkpoint_freq = args["logging"].get(
        "checkpoint_freq", 100
    )  # get default frequency, default to 100 otherwise
    logging_frequency = args["logging"].get("logging_frequency", 3)  # default to 3
    output_file = args["logging"].get("output_file", tag)
    plot_matrices = args["logging"].get("plot_matrices", True)
    tensorboard_dir = folder  # args['logging'].get('tensorboard_dir', 'runs/')
    use_tensorboard = args["logging"].get("use_tensorboard", False)

    # force_cudnn_initialization()
    dump = os.path.join(folder, "params-ijepa.yaml")
    with open(dump, "w") as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    if rank > 0:
        logger.setLevel(logging.ERROR)
    logger.info(f"train.py: {_GLOBAL_SEED=}")  # log seed
    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    output_file = os.path.join(folder, output_file)
    # tensorboard_dir = os.path.join(folder, tensorboard_dir)
    logger.addHandler(logging.FileHandler(output_file))  # add auto output ;)

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
        tensorboard_dir=tensorboard_dir,
    )
    import re

    match_ = re.search(r"ep(\d+)", r_file)  # extract the number based on the checkpoint
    pretrain_epoch = match_.group(1)
    # pretrain_epoch = 1000
    target_encoder = copy.deepcopy(encoder)
    if use_tensorboard:
        encoder.init_summary_writer("context_encoder", pretrain_epoch)
        target_encoder.init_summary_writer("target_encoder", pretrain_epoch)
        predictor.init_summary_writer("predictor", pretrain_epoch)

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
        min_keep=min_keep,
    )

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
    )

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
    )
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
        use_bfloat16=use_bfloat16,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = (
            load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
            )
        )
        # for _ in range(start_epoch*ipe):
        #     scheduler.step()
        #     wd_scheduler.step()
        #     next(momentum_scheduler)
        #     mask_collator.step()

    # -- TRAINING LOOP
    start_time = time.perf_counter()  # get starting time
    # for epoch in range(start_epoch, num_epochs):
    start_time_epoch = time.perf_counter()
    logger.info("Starting")

    all_model_sims, all_target_sims, cross_sims = [], [], []
    all_z = []
    all_h = []
    all_patch_losses = []
    all_patch_grad_norms = []
    sensitivity_rows = []

    num_eval_batches = int(args.get("evaluation", {}).get("num_eval_batches", 100))
    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
        progress_freq = max(1, num_eval_batches // 10)
        if itr % progress_freq == 0:
            logger.info("Iteration: %d" % itr)
        if itr >= num_eval_batches:
            break

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
                # logger.critical('z[:100] values: %s' % (str(z[:100])))
                return z

            def loss_fn(z, h):
                # this should be fully functional, as proven by L2
                final_loss = which_loss.__dict__[loss_function](z, h)
                # loss_l2 = F.smooth_l1_loss(z, h) # initial loss
                loss = AllReduce.apply(final_loss)
                return loss

            # Step 1. Forward
            with torch.no_grad():
                h_unflat = forward_target()
                z_unflat = forward_context()

                z = z_unflat.reshape(-1, z_unflat.size(-1))
                h = h_unflat.reshape(-1, h_unflat.size(-1))

                # Existing per-patch diagnostics on the clean targets.
                per_patch_mse = per_patch_objective_loss(z, h, loss_function)
                D = z.size(-1)
                per_patch_grad_norm = (2.0 / D) * torch.norm(z - h, p=2, dim=-1)

                # PDF sensitivity probe: corrupt targets only, keep predictions fixed.
                batch_sensitivity_rows = []
                if run_sensitivity:
                    for ratio in corruption_ratios:
                        h_corrupt_unflat, corruption_mask = corrupt_target_representations(
                            h=h_unflat,
                            corruption_ratio=ratio,
                            batch_size=imgs.size(0),
                        )

                        h_corrupt = h_corrupt_unflat.reshape(-1, h_corrupt_unflat.size(-1))

                        row = {'corruption_ratio': float(ratio)}

                        for objective_name in sensitivity_objectives:
                            clean_patch_loss = objective_loss_per_patch(z, h, objective_name)
                            corrupted_patch_loss = objective_loss_per_patch(z, h_corrupt, objective_name)

                            stats = summarize_per_patch_sensitivity(
                                clean_patch_loss=clean_patch_loss,
                                corrupted_patch_loss=corrupted_patch_loss,
                                corruption_mask=corruption_mask,
                            )

                            for k, v in stats.items():
                                row[f'{objective_name}_{k}'] = v

                        batch_sensitivity_rows.append(row)

            return per_patch_mse, per_patch_grad_norm, batch_sensitivity_rows

        # (model_sim, target_sim, cross_sim), etime = gpu_timer(train_step)
        (per_patch_mse, per_patch_grad_norm, batch_sensitivity_rows), etime = gpu_timer(
            train_step
        )

        # all_model_sims. append(model_sim.detach().cpu().numpy())
        # all_target_sims.append(target_sim.detach().cpu().numpy())
        # cross_sims.append(cross_sim.detach().cpu().numpy())
        all_patch_losses.append(per_patch_mse.reshape(-1).detach().cpu())
        all_patch_grad_norms.append(per_patch_grad_norm.detach().cpu())
        sensitivity_rows.extend(batch_sensitivity_rows)
        # difference = difference.detach().cpu().numpy()
        # loss = loss.detach().cpu().numpy()

    # logger.info('All model similarities: %s', str(all_model_sims[:5000]))
    # logger.info('All target similarities: %s',str(all_target_sims[:5000]))

    # after all iterations
    # save_checkpoint(epoch+1)

    # -- Visualize weights using Summary Writer - old method with no filtering#
    # ep = '-ep100'
    if 1 == 0:
        import re

        match_ = re.search(
            r"ep(\d+)", r_file
        )  # extract the number based on the checkpoint
        ep = match_.group(1)
        writer = SummaryWriter(f"runs/l2-{ep}")
        # logger.critical(str(len(all_model_sims)))
        all_params = []
        for idx, (name, param) in enumerate(encoder.named_parameters()):
            # logger.info('epoch: %s, name: %s, param: %s '
            #             % (ep, name, param))
            # logger.info('extending params epoch: %s, name: %s' % (ep, name))
            all_params.extend(param.view(-1).detach().cpu().numpy())
            writer.add_histogram(name, param, global_step=idx, bins=1000)
    """
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
    """
    # Z = torch.cat(all_z, dim=0)
    # H = torch.cat(all_h, dim=0)
    # logger.info(f'Z shape: {Z.shape}')

    # z_metrics = representation_metrics(Z)
    # h_metrics = representation_metrics(H)

    # logger.info(
    #     f"Context Encoder:"
    #     f" VNE={z_metrics['vne']:.4f}"
    #     f" NormVNE={z_metrics['vne_norm']:.4f}"
    #     f" EffRank={z_metrics['effective_rank']:.4f}"
    #     f" NormEffRank={z_metrics['effective_rank_norm']:.4f}"
    # )

    # logger.info(
    #     f"Target Encoder:"
    #     f" VNE={h_metrics['vne']:.4f}"
    #     f" NormVNE={h_metrics['vne_norm']:.4f}"
    #     f" EffRank={h_metrics['effective_rank']:.4f}"
    #     f" NormEffRank={h_metrics['effective_rank_norm']:.4f}"
    # )

    if run_sensitivity and len(sensitivity_rows) > 0:
        import csv
        from collections import defaultdict
        import numpy as np
        per_patch_metric_names = [
            "all_clean_mean",
            "all_corrupted_mean",
            "all_abs_delta_mean",
            "all_rel_delta_mean",
            "all_rel_delta_median",
            "all_rel_delta_p95",
            "all_rel_delta_p99",
            "corrupted_only_abs_delta_mean",
            "corrupted_only_abs_delta_median",
            "corrupted_only_abs_delta_p95",
            "corrupted_only_abs_delta_p99",
            "corrupted_only_rel_delta_mean",
            "corrupted_only_rel_delta_median",
            "corrupted_only_rel_delta_p95",
            "corrupted_only_rel_delta_p99",
            "num_patches",
            "num_corrupted_patches",
        ]
        grouped = defaultdict(list)
        for row in sensitivity_rows:
            grouped[row["corruption_ratio"]].append(row)

        summary_rows = []
        for ratio in sorted(grouped.keys()):
            rows = grouped[ratio]
            summary = {"corruption_ratio": ratio}
            for objective_name in sensitivity_objectives:
                for metric_name in per_patch_metric_names:
                    key = f'{objective_name}_{metric_name}'
                    vals = [r[key] for r in rows if key in r]

                    if key.endswith("num_patches") or key.endswith("num_corrupted_patches"):
                        summary[key] = int(np.sum(vals)) if vals else 0
                    else:
                        summary[key] = float(np.mean(vals)) if vals else float('nan')
            summary_rows.append(summary)

        logger.info("Target-representation corruption sensitivity:")

        

        header_parts = ["corruption_ratio"]
        
        for objective_name in sensitivity_objectives:
            for metric_name in per_patch_metric_names:
                header_parts.append(f"{objective_name}_{metric_name}")
        logger.info(", ".join(header_parts))
        for row in summary_rows:
            logger.info(
                ", ".join(
                    f"{row[k]:.8e}" if isinstance(row[k], float) else str(row[k])
                    for k in header_parts
                )
            )

        csv_path = os.path.join(
            folder, f"target-corruption-sensitivity-ep{pretrain_epoch}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header_parts)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Saved target corruption sensitivity CSV to: {csv_path}")

        if plot_matrices:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            x_values = [100.0 * row["corruption_ratio"] for row in summary_rows]
            for objective_name in sensitivity_objectives:
                y_values = [
                    row[f"{objective_name}_relative_increase"] for row in summary_rows
                ]
                ax.plot(
                    x_values, y_values, marker="o", linewidth=1.5, label=objective_name
                )
            ax.set_title(
                f"Target-representation outlier sensitivity, epoch {pretrain_epoch}"
            )
            ax.set_xlabel("Corrupted target representations (%)")
            ax.set_ylabel("Relative loss increase")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plot_path = os.path.join(
                folder, f"target-corruption-sensitivity-ep{pretrain_epoch}.png"
            )
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved target corruption sensitivity plot to: {plot_path}")

    all_patch_losses = torch.cat(all_patch_losses, dim=0)
    # all_patch_losses = torch.tensor(all_patch_losses)
    all_patch_grad_norms = torch.cat(all_patch_grad_norms, dim=0)

    tail_metrics = compute_tail_metrics(all_patch_losses)
    grad_tail_metrics = compute_tail_metrics(all_patch_grad_norms)

    logger.info("Per-patch MSE tail analysis:")
    for k, v in tail_metrics.items():
        logger.info(f"{k}: {v}")

    logger.info("\n")
    logger.info("Per-patch representation-gradient-norm tail analysis:")
    for k, v in grad_tail_metrics.items():
        logger.info(f"grad_norm_{k}: {v}")

    if plot_matrices:
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------
        # Plot ordered per-patch losses.
        #
        # Option A, active:
        #   x-axis = individual patches ordered by descending loss
        #   y-axis = per-patch loss
        #
        # Option B, commented:
        #   x-axis = individual patches ordered by descending loss
        #   y-axis = cumulative loss up to that patch
        # ------------------------------------------------------------
        # Batch-level alternative:
        # To plot batches instead of individual patches, store one scalar per batch
        # during the evaluation loop, e.g.:
        #
        #     batch_loss = per_patch_mse.sum()
        #     all_batch_losses.append(batch_loss.detach().cpu())
        #
        # Then after the loop:
        #
        #     all_batch_losses = torch.stack(all_batch_losses)
        #     sorted_batch_losses, _ = torch.sort(all_batch_losses, descending=True)
        #     y_values = sorted_batch_losses.numpy()
        #     x_values = np.arange(1, len(y_values) + 1)
        sorted_patch_losses, _ = torch.sort(all_patch_losses, descending=True)

        # Option A: raw sorted per-patch losses
        y_values = sorted_patch_losses.numpy()
        ylabel = "Per-patch loss"

        # Option B: cumulative sorted loss
        # Uncomment these two lines instead of Option A if needed.
        # y_values = torch.cumsum(sorted_patch_losses, dim=0).numpy()
        # ylabel = "Cumulative MSE loss"

        x_values = np.arange(1, len(y_values) + 1)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        ax.plot(x_values, y_values, linewidth=1.0)

        ax.set_title(f"Ordered per-patch loss, epoch {pretrain_epoch}")
        ax.set_xlabel("Patch rank, sorted by descending loss")
        ax.set_ylabel(ylabel)

        ax.grid(True, alpha=0.3)

        outfile = os.path.join(folder, f"ordered-loss-ep{pretrain_epoch}.png")
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ordered loss plot to: {outfile}")

    if plot_matrices and 1 == 0:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        im = ax.imshow(difference, interpolation="nearest", vmax=0.599)
        # plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(
            "Context-Target Gram Matrix Divergence", rotation=90, fontsize=14
        )
        cbar.ax.tick_params(labelsize=10)

        # ax.set_title(f'KL divergence: {loss}')

        # --- Region definition ---
        x, y = 5, 38
        w, h = 6, 6

        # --- Main red rectangle ---
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # --- Zoomed inset (bigger + outside) ---
        axins = zoomed_inset_axes(
            ax,
            zoom=6,
            loc="upper left",
            bbox_to_anchor=(-0.7, 1),  # push outside
            bbox_transform=ax.transAxes,
        )

        axins.imshow(difference, interpolation="nearest", vmax=0.599)
        import numpy as np

        sub = difference[y : y + h, x : x + w]

        for i in range(1, h):
            for j in range(1, w):
                val = sub[i, j]
                axins.text(
                    j + x,  # center of cell (x coord in data space)
                    i + y,  # center of cell (y coord in data space)
                    f"{val:.2f}",  # format (adjust precision if needed)
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white" if val < 0.3 else "black",  # contrast heuristic
                )
        # Limit inset to the selected region
        axins.set_xlim(x, x + w)
        axins.set_ylim(y + h, y)  # invert y-axis for imshow

        axins.set_xticks([])
        axins.set_yticks([])

        # Style inset border
        for spine in axins.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)

        # --- Connect the two rectangles ---
        # mark_inset(ax, axins, loc1=4, loc2=4, fc="none", ec="red", linewidth=1.5)
        from matplotlib.patches import ConnectionPatch

        con1 = ConnectionPatch(
            xyA=(x, y),
            coordsA=ax.transData,
            xyB=(1, 1),
            coordsB=axins.transAxes,
            color="red",
            linewidth=1.5,
        )

        con2 = ConnectionPatch(
            xyA=(x, y + h),
            coordsA=ax.transData,
            xyB=(1, 0),
            coordsB=axins.transAxes,
            color="red",
            linewidth=1.5,
        )

        fig.add_artist(con1)
        fig.add_artist(con2)

        # Save
        outfile = os.path.join(folder, f"difference-matrix-ep{pretrain_epoch}.png")
        plt.savefig(outfile, bbox_inches="tight")  # important when placing outside
        plt.close()

    if plot_matrices and 1 == 0:
        all_model_sims = np.array(all_model_sims)
        all_target_sims = np.array(all_target_sims)
        logger.critical(str("shape of all model sims" + all_model_sims.shape))
        all_model_sims = all_model_sims.mean()  # extact mean
        all_target_sims = all_target_sims.mean()
        # i guess the shape is []
        ri = torch.randint(
            0, len(all_model_sims), (1,)
        )  # pick a random image from the batch
        lb = min(
            all_model_sims[ri].min(), all_target_sims[ri].min(), cross_sims[ri].min()
        )  # lower bound
        ub = max(
            all_model_sims[ri].max(), all_target_sims[ri].max(), cross_sims[ri].max()
        )  # upper bound
        lb = 0.0
        ub = 1.0
        fig = plt.figure(figsize=(21, 7), dpi=300)
        data = (all_model_sims[ri], all_target_sims[ri], cross_sims[ri])
        titles = ("Predictions", "Targets", "Cross")
        for idx, datum in enumerate(data):
            plt.subplot(1, 2, idx + 1)
            img = plt.imshow(datum, interpolation="nearest", vmin=lb, vmax=ub)
            plt.colorbar(img, fraction=0.046, pad=0.04)
            plt.title(titles[idx])
        plt.show()
        plt.suptitle("Evaluation of %s" % (r_file))
        # outfile = os.path.join(folder, f'sims-PKT-ep{ep}.png')
        outfile = os.path.join(folder, f"sim_matrices_avg{ep}.png")
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
    logger.info("time taken for epoch %s" % str(datetime.timedelta(seconds=time_epoch)))

    total_time = time.perf_counter() - start_time
    logger.info(
        "Total pretraining time %s" % str(datetime.timedelta(seconds=total_time))
    )


if __name__ == "__main__":
    main()
