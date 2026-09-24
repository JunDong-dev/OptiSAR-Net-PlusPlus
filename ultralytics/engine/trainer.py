# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import csv
import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)


class BaseTrainer:
    """
    Base class for trainers.

    Attributes:
        args (SimpleNamespace): configuration for the trainer.
        validator (BaseValidator): validator instance.
        model (nn.Module): model instance.
        callbacks (defaultdict): dict of callbacks.
        save_dir (Path): directory to save results.
        wdir (Path): directory to save weights.
        last (Path): path to the latest checkpoint.
        best (Path): path to the best checkpoint.
        save_period (int): save a checkpoint every x epochs (disabled if < 1).
        batch_size (int): training batch size.
        epochs (int): total number of training epochs.
        start_epoch (int): starting epoch for training.
        device (torch.device): device used for training.
        amp (bool): flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): GradScaler for AMP.
        data (str): data path.
        trainset (torch.utils.data.Dataset): training dataset.
        testset (torch.utils.data.Dataset): test dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): resume training from a checkpoint.
        lf (nn.Module): loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
        best_fitness (float): best fitness value achieved.
        fitness (float): current fitness value.
        loss (float): current loss value.
        tloss (float): total loss value.
        loss_names (list): list of loss names.
        csv (Path): path to the results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): path to a config file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Directory setup
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for logger
        self.wdir = self.save_dir / "weights"  # weights directory
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make directory
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run arguments
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # guard against users accidentally passing epochs=None
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device setup
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster on CPU; time is spent on inference rather than data loading

        # Model and dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading the dataset multiple times
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimizer utilities
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Append the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='' or device=None on multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is a list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess for DDP training; otherwise train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning("WARNING ⚠️ AutoBatch 'batch<1' is incompatible with multi-GPU training, using default 'batch=16'")
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize the learning rate scheduler for training."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initialize and set the DistributedDataParallel training parameters."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to force timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only float tensors can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer to customize frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # single GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks
        self.amp = bool(self.amp)  # convert to boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # used for multi-scale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single GPU only, estimate the best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        if batch_size <= 0:
            raise ValueError(
                f"Computed batch_size={batch_size} is invalid. "
                f"Total batch_size={self.batch_size}, world_size={world_size}. "
                "Ensure total batch_size >= world_size, or reduce the number of GPUs used."
            )
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: doubling the batch size for DOTA datasets may OOM on images with >2000 objects
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.val_event_best_fitness = float("-inf")
        self.val_event_non_improve_streak = 0
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train the model, then evaluate and plot as specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure a stable training start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interpolation
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Logging
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    # Get the current learning rate (of the primary parameter group)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (3 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory usage
                            current_lr,  # current learning rate
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e. 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # Update the end-to-end loss schedule when enabled.
            if hasattr(self.model, "criterion") and hasattr(self.model.criterion, "update"):
                self.model.criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for logger
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                validated_this_epoch = False
                event_based_stopping = self.args.val_event_patience > 0
                scheduled_validation = (epoch + 1) % self.args.val_interval == 0
                fallback_validation = (
                    (self.epochs - epoch <= 5) or self.stopper.possible_stop or self.stop
                )
                if self.args.val and (
                    scheduled_validation
                    or final_epoch
                    or (fallback_validation and not event_based_stopping)
                ):
                    self.metrics, self.fitness = self.validate()
                    validated_this_epoch = True
                    self.stop |= self._update_val_event_early_stopping(self.fitness)
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model(save_best=validated_this_epoch)
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if epochs exceeded
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early stopping
            if RANK != -1:  # DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Final validation with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Get the batch size by computing the model's memory footprint."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # return batch size

    def _get_memory(self):
        """Get accelerator memory usage in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scoped to speed up 'import ultralytics'

        return pd.read_csv(self.csv).to_dict(orient="list")

    def _model_train(self):
        """Set the model to training mode."""
        self.model.train()
        # Freeze BatchNorm
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self, save_csv=True, save_best=False):
        """Save a model training checkpoint with attached metadata."""
        import io

        # Serialize the checkpoint to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                # Keep EMA weights and buffers in FP32. Casting the complete module to FP16 can overflow
                # large BatchNorm running variances to +Inf and make the saved checkpoint differ from
                # the in-memory model used for validation.
                "ema": deepcopy(self.ema.ema).float(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as a dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv() if save_csv and self.csv.exists() else {},
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if (
            save_best
            and self.best_fitness is not None
            and self.fitness is not None
            and self.best_fitness == self.fitness
        ):
            self.best.write_bytes(serialized_ckpt)  # save best.pt
            LOGGER.info(f"💾 Saving best.pt with meanIoU (fitness) = {self.fitness:.6f} (epoch {self.epoch})")
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
            (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Extract train/val paths from the data dict (if present).

        Returns None if the data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download a model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if the model is already loaded; no setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing of model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Run validation on the test set using self.validator.

        The returned dict should contain a "fitness" key.
        """
        metrics = self.validator(self)
        # If fitness is not found, fall back to the loss (if present)
        if "fitness" not in metrics:
            if self.loss is not None:
                fitness = -self.loss.detach().cpu().numpy()
            else:
                fitness = 0.0  # last-resort fallback
                LOGGER.warning("No fitness found in metrics and loss is None, using 0.0 as default")
        else:
            fitness = metrics.pop("fitness")

        if not self.best_fitness or self.best_fitness < fitness:
            old_best = self.best_fitness if self.best_fitness is not None else 0.0
            self.best_fitness = fitness
            LOGGER.info(
                f"🏆 New best meanIoU (fitness) found: {fitness:.6f} "
                f"(previous best: {old_best:.6f}, improvement: {fitness - old_best:.6f})"
            )
        return metrics, fitness

    def _update_val_event_early_stopping(self, fitness):
        """Stop after a configured number of validation events fail to improve fitness."""
        patience = int(getattr(self.args, "val_event_patience", 0))
        if patience <= 0 or fitness is None:
            return False

        if fitness > self.val_event_best_fitness:
            improvement = fitness - self.val_event_best_fitness
            self.val_event_best_fitness = fitness
            self.val_event_non_improve_streak = 0
            LOGGER.info(
                f"Validation-event early stopping: new best meanIoU={fitness:.6f} "
                f"(improvement={improvement:.6f}, streak reset)."
            )
            return False

        self.val_event_non_improve_streak += 1
        LOGGER.info(
            f"Validation-event early stopping: meanIoU={fitness:.6f} did not exceed "
            f"best={self.val_event_best_fitness:.6f} "
            f"({self.val_event_non_improve_streak}/{patience} non-improving validations)."
        )
        if self.val_event_non_improve_streak >= patience:
            LOGGER.info(
                "Validation-event early stopping: stopping because the configured number "
                "of consecutive validation events did not improve meanIoU."
            )
            return True
        return False

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get the model and raise NotImplementedError for loading a cfg file."""
        raise NotImplementedError("Loading a cfg file is not supported by this task trainer")

    def get_validator(self):
        """Return NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function is not implemented in the trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return a dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function is not implemented in the trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build a dataset."""
        raise NotImplementedError("build_dataset function is not implemented in the trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labeled training loss items tensors.

        Note:
            Not necessary for classification but essential for segmentation and detection.
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO models."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: the following functions may need to be moved to callbacks
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO models."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        metric_keys = list(metrics)
        fieldnames = ["epoch", "time", *metric_keys]
        existing_rows = []

        if self.csv.exists():
            with self.csv.open(newline="") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)
            new_keys = [key for key in metric_keys if key not in existing_fieldnames]
            fieldnames = [*existing_fieldnames, *new_keys]
            if new_keys:
                temporary_csv = self.csv.with_suffix(".csv.tmp")
                with temporary_csv.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                os.replace(temporary_csv, self.csv)

        def format_value(value):
            try:
                return f"{value:.6g}"
            except (TypeError, ValueError):
                return value

        row = {
            "epoch": self.epoch + 1,
            "time": format_value(time.time() - self.train_time_start),
            **{key: format_value(value) for key, value in metrics.items()},
        }
        write_header = not self.csv.exists()
        with self.csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot_metrics(self):
        """Plot and visualize metrics."""
        pass

    def on_plot(self, name, data=None):
        """Register plots (e.g. for callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Run final evaluation and validation on the object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if a resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check the resume data YAML exists, otherwise strip it to force re-download of the dataset
                ckpt_args = attempt_load_weights(last).args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # resume model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow argument updates to reduce memory or update the device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from a given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. "
                f"Fine-tuning for {self.epochs} additional epochs."
            )
            self.epochs += ckpt["epoch"]  # fine-tune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build an optimizer with separate decay, normalization, and bias parameter groups."""
        g = [[], [], []]
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' selected; ignoring "
                f"'lr0={self.args.lr0}' and 'momentum={self.args.momentum}'."
            )
            nc = getattr(model, "nc", 10)
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10_000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1].append(param)
                else:
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' is not supported. Available options: {sorted(optimizers)}")

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], **optim_args)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], **optim_args)
        else:
            optimizer = optim.SGD(g[2], **optim_args)

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with "
            f"{len(g[1])} normalization weights, {len(g[0])} decayed weights, and {len(g[2])} biases"
        )

        return optimizer
