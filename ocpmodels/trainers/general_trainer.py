import datetime
import logging
import os
import subprocess
import pickle
from typing import Optional, List

import numpy as np
import torch
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
import torch_geometric
from tqdm import tqdm

import ocpmodels

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollaterMultimodal, BalancedBatchSampler
from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.common.utils import save_checkpoint

from ocpmodels.modules.exponential_moving_average import ExponentialMovingAverage
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.torchmetrics_evaluator import convert_nested_dict_tensor, combine_dict
from ocpmodels.modules import task_config

from ocpmodels.trainers import BaseTrainer


@registry.register_trainer("general")
class GeneralTrainer(BaseTrainer):
    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir=None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed=None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        name: str = "base_trainer",
        slurm={},
        noddp: bool = False,
        config: dict = {},
    ) -> None:
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0

        if not identifier:
            identifier = config["task"].get("identifier", None)

        self.device: torch.device
        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        self.timestamp_id: str
        if not timestamp_id:
            timestamp_id = config["task"].get("timestamp_id", None)
        if timestamp_id is None:
            timestamp = torch.tensor(datetime.datetime.now().timestamp()).to(
                self.device
            )
            # create directories from master rank only
            distutils.broadcast(timestamp, 0)
            timestamp = datetime.datetime.fromtimestamp(
                timestamp.float().item()
            ).strftime("%Y-%m-%d-%H-%M-%S")
            if identifier:
                self.timestamp_id = f"{timestamp}-{identifier}"
            else:
                self.timestamp_id = timestamp
        else:
            self.timestamp_id = timestamp_id

        try:
            commit_hash = (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        assert_is_instance(ocpmodels.__path__[0], str),
                        "describe",
                        "--always",
                    ]
                )
                .strip()
                .decode("ascii")
            )
        # catch instances where code is not being run from a git repo
        except Exception:
            commit_hash = None

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "trainer": "general",
            "model": model,
            "optim": optimizer,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(
                    run_dir, "results", self.timestamp_id
                ),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "noddp": noddp,
        }
        self.config.update({key: config[key] for key in config if key not in self.config})
        # AMP Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "%s_%s" % (
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"][
                "folder"
            ].replace("%j", self.config["slurm"]["job_id"])
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            self.config["dataset"] = dataset.get("train", None)
            self.config["val_dataset"] = dataset.get("val", None)
            self.config["test_dataset"] = dataset.get("test", None)
        else:
            self.config["dataset"] = dataset

        if not is_debug and distutils.is_master() and not is_hpo:
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        self.is_debug = is_debug
        self.is_hpo = is_hpo

        if self.is_hpo:
            # conditional import is necessary for checkpointing

            # sets the hpo checkpoint frequency
            # default is no checkpointing
            self.hpo_checkpoint_every = self.config["optim"].get(
                "checkpoint_every", -1
            )

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))
        self.load()

        self.evaluator = getattr(task_config, self.config["task"]["evaluators"])(
            self.config["evaluators"], sync_on_compute=False).to(self.device)
        self.loss_logger = getattr(task_config, self.config["task"]["loss_loggers"])(
            self.config["loss_loggers"], sync_on_compute=False).to(self.device)
        self.primary_metric = self.config["task"]["primary_metric"]
        self.primary_metric_max = self.config["task"].get("primary_metric_max", True)

    def compare_op(self, a, b):
        if self.primary_metric_max:
            return a > b
        else:
            return a < b

    def get_dataloader(self, dataset, sampler, **kargs) -> DataLoader:
        loader = DataLoader(
            dataset,
            collate_fn=self.parallel_collater,
            num_workers=self.config["task"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
            **kargs,
        )
        return loader

    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool, drop_last: bool = False,
    ) -> BalancedBatchSampler:
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
            drop_last=drop_last,
        )
        return sampler

    def load_datasets(self) -> None:
        self.parallel_collater = ParallelCollaterMultimodal(
            0 if self.cpu else 1,
            self.config["model"]["attributes"].get("otf_graph", False),
            self.config["model"].get("tokenizer",  None),
        )

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.config.get("dataset", None):
            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["task"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            if self.config.get("val_dataset", None):
                self.val_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["val_dataset"])
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config["task"].get(
                        "eval_batch_size", self.config["task"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            if self.config.get("test_dataset", None):
                self.test_dataset = registry.get_dataset_class(
                    self.config["task"]["dataset"]
                )(self.config["test_dataset"])
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config["task"].get(
                        "eval_batch_size", self.config["task"]["batch_size"]
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.config["task"].get("normalizers", None):
            self.normalizers = getattr(task_config, self.config["task"]["normalizers"])(
                self.config["normalizers"], self.train_loader, self.device)

    def load_task(self) -> None:
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.task_names = [task["name"] for task in self.config["task"]["tasks"]]
        self.task_weights = {task["name"]: task.get("weight", 1.0) for task in self.config["task"]["tasks"]}

    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        self.model = registry.get_model_class(self.config["model"]["name"])(
            self.config["model"]['attributes'],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device], find_unused_parameters=True,
            )

    def load_loss(self) -> None:
        self.loss_fn = getattr(task_config, self.config["task"]["losses"])(
            self.config["losses"]
        )

    def load_optimizer(self) -> None:
        self.optimizer = getattr(task_config, self.config["task"]["optim"])(
            self.config["optim"],
            self.model
        )

    def load_extras(self) -> None:
        if ((self.config["task"].get("scheduler", None) is not None) and
                (self.config.get("scheduler", None) is not None)):
            self.scheduler = getattr(task_config, self.config["task"]["scheduler"])(
                self.optimizer,
                self.config["scheduler"],
            )
        else:
            self.scheduler = None
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
            self,
            metrics=None,
            checkpoint_file: str = "checkpoint.pt",
            training_state: bool = True,
    ):
        if not self.is_debug and distutils.is_master():
            if training_state:
                return save_checkpoint(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.scheduler.state_dict()
                        if self.scheduler.scheduler_type != "Null"
                        else None,
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.primary_metric,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
                return ckpt_path
        return None

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["task"].get(
            "eval_every", len(self.train_loader)
        )

        print_every = self.config["task"].get(
            "print_every", 10
        )

        self.metric = {}

        if self.primary_metric_max:
            self.best_val_metric = -np.inf
        else:
            self.best_val_metric = np.inf

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["task"]["max_epochs"]
        ):
            # self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch, self.loss_logger)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self._compute_metrics(
                    out,
                    batch,
                    self.evaluator
                )

                if (
                    self.step % print_every == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    metrics = self.evaluator.compute()
                    loss_meter = self.loss_logger.compute()
                    self.metrics = convert_nested_dict_tensor(combine_dict(metrics, loss_meter))

                    # Log metrics.
                    log_dict = self.metrics.copy()
                    log_dict.update(
                        {
                            "lr": self.scheduler.get_lr(),
                            "epoch": self.epoch,
                            "step": self.step,
                        }
                    )

                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.evaluator.reset()
                    self.loss_logger.reset()

                    if self.logger is not None:
                        self.logger.log(
                            log_dict,
                            step=self.step,
                            split="train",
                        )

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if self.compare_op(val_metrics[self.primary_metric], self.best_val_metric):
                            self.best_val_metric = val_metrics[self.primary_metric]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=False,
                                )

                        if self.is_hpo:
                            metrics = self.evaluator.compute()
                            loss_logger = self.loss_logger.compute()
                            loss_logger = {key: {loss_logger[key][key1] / scale for key1 in loss_logger[key]}
                                           for key in loss_logger}
                            self.metrics = convert_nested_dict_tensor(combine_dict(metrics, loss_logger))
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler:
                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        if self.step % eval_every == 0:
                            self.scheduler.step(
                                metrics=val_metrics[self.primary_metric],
                            )
                    else:
                        self.scheduler.step()

                # # debug of torchmetrics in distributed training
                # break

            torch.cuda.empty_cache()
            self.evaluator.reset()
            self.loss_logger.reset()

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()


    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator = getattr(task_config, self.config["task"]["evaluators"])(
            self.config["evaluators"]).to(self.device)
        loss_logger = getattr(task_config, self.config["task"]["loss_loggers"])(
            self.config["loss_loggers"]).to(self.device)
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch, loss_logger)

            # Compute metrics.
            self._compute_metrics(out, batch, evaluator)

            # # debug of torchmetrics in distributed training
            # break

        metrics = evaluator.compute()
        loss_meter = loss_logger.compute()
        evaluator.reset()
        loss_logger.reset()
        metrics = convert_nested_dict_tensor(combine_dict(metrics, loss_meter))
        log_dict = metrics.copy()
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    def _forward(self, batch_list):
        output = self.model(batch_list)
        if not isinstance(output, dict):
            output = {task: output for task in self.task_names}

        return output

    def _compute_loss(self, out, batch_list, loss_logger):
        target = {
            task: torch.cat([batch[task].to(self.device) for batch in batch_list], dim=0)
            for task in self.task_names
        }

        target_normed = {
            task: self.normalizers[task].norm(target[task])
            if self.normalizers.get(task, None) is not None else target[task]
            for task in self.task_names
        }

        loss = {
            task: self.loss_fn[task](out[task], target_normed[task])
            for task in self.task_names
        }

        if len(self.task_names) > 1:
            loss["total_loss"] = sum([loss[task] * self.task_weights[task] for task in loss])
            result = loss["total_loss"]
        else:
            result = loss[self.task_names[0]]

        # Sanity check to make sure the compute graph is correct.
        for key in loss:
            assert hasattr(loss[key], "grad_fn")

        num_samples = out[self.task_names[0]].shape[0]
        weight = {key: num_samples for key in loss}
        loss_logger.update(loss, weight=weight)

        # debug of torchmetrics in distributed training
        # total 6 samples, 3 for each rank, total weighted mean = (1*1 + 4*2 + 8*3 + 10*2) / (1 + 4 + 8 + 2) = 3.5
        # if distutils.is_master():
        #     target = {task: torch.tensor([1, 2, 3], device=self.device)
        #               for task in self.task_names}
        #     weights = {task: torch.tensor([1, 4, 8], device=self.device)
        #              for task in self.task_names}
        #     metrics = loss_logger(target, weights)
        #     # weighted mean = (1*1 + 4*2 + 8*3) / (1 + 4 + 8) = 2.5385
        #     print('step', distutils.get_rank(), metrics)
        # else:
        #     target = {task: torch.tensor(10, device=self.device)
        #               for task in self.task_names}
        #     weights = {task: torch.tensor(2, device=self.device)
        #                for task in self.task_names}
        #     metrics = loss_logger(target, weights)
        #     # weighted mean = (10*2) / (2) = 10
        #     print('step', distutils.get_rank(), metrics)

        return result

    def _compute_metrics(self, out, batch_list, evaluator):
        target = {
            task: torch.cat([batch[task].to(self.device) for batch in batch_list], dim=0)
            for task in self.task_names
        }

        out = {
            task: self.normalizers[task].denorm(out[task])
            if self.normalizers.get(task, None) is not None else out[task]
            for task in self.task_names
        }

        metrics = evaluator(out, target)

        # debug of torchmetrics in distributed training
        # total 6 samples, 3 for each rank, total accuracy = 5/6 = 0.8333
        # if distutils.is_master():
        #     target = {task: torch.tensor([1, 1, 1], device=self.device)
        #               for task in self.task_names}
        #     preds = {task: torch.tensor([[0.1, 0.9], [0.3, 0.1], [0.2, 0.5]], device=self.device)
        #              for task in self.task_names}
        #     metrics = evaluator(preds, target)
        #     # accuracy = 2/3 = 0.6667
        #     print('step', distutils.get_rank(), metrics)
        # else:
        #     target = {task: torch.tensor([1, 0, 1], device=self.device)
        #               for task in self.task_names}
        #     preds = {task: torch.tensor([[0.1, 0.9], [0.3, 0.1], [0.2, 0.5]], device=self.device)
        #              for task in self.task_names}
        #     metrics = evaluator(preds, target)
        #     # accuracy = 3/3 = 1
        #     print('step', distutils.get_rank(), metrics)

        return metrics

    @torch.no_grad()
    def predict(
        self,
        loader,
        results_file=None,
        disable_tqdm: bool = False,
        other_output_keys_in_prediction: List = [],
    ):
        ensure_fitted(self._unwrapped_model)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        for task in self.normalizers:
            if self.normalizers.get(task, None) is not None:
                self.normalizers[task].to(self.device)

        predictions = {"id": []}
        predictions.update({task: [] for task in self.task_names})
        predictions.update({key: [] for key in other_output_keys_in_prediction})

        for _, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            for task in self.normalizers:
                if self.normalizers.get(task, None) is not None:
                    out[task] = self.normalizers[task].denorm(
                        out[task])

            try:
                predictions["id"].extend(
                    [str(i) for i in batch[0].sid.tolist()]
                )
            except:
                predictions["id"].extend(
                    [str(i) for i in batch[0].sid]
                )
            for task in self.task_names:
                predictions[task].extend(
                    out[task].cpu().detach().numpy()
                )
            for key in other_output_keys_in_prediction:
                predictions[key].extend(
                    torch.cat(
                        [batch_item[key] for batch_item in batch], dim=0
                    ).cpu().detach().numpy()
                )

        self.save_results(
            predictions,
            results_file,
            keys=self.task_names + other_output_keys_in_prediction
        )

        if self.ema:
            self.ema.restore()

        return predictions

    @torch.no_grad()
    def predict_async(
        self,
        loader,
        results_file=None,
        disable_tqdm: bool = False,
        per_image: bool = False,
        other_output_keys_in_prediction: List = [],
    ):
        def run_predict(loader):
            for _, batch in tqdm(
                    enumerate(loader),
                    total=len(loader),
                    position=rank,
                    desc="device {}".format(rank),
                    disable=disable_tqdm,
            ):
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)

                for task in self.normalizers:
                    if self.normalizers.get(task, None) is not None:
                        out[task] = self.normalizers[task].denorm(
                            out[task])

                try:
                    predictions["id"].extend(
                        [str(i) for i in batch[0].sid.tolist()]
                    )
                except:
                    predictions["id"].extend(
                        [str(i) for i in batch[0].sid]
                    )
                for task in self.task_names:
                    predictions[task].extend(
                        out[task].cpu().detach().numpy()
                    )
                for key in other_output_keys_in_prediction:
                    predictions[key].extend(
                        torch.cat(
                            [batch_item[key] for batch_item in batch], dim=0
                        ).cpu().detach().numpy()
                    )

        ensure_fitted(self._unwrapped_model)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]
        else:
            test_sampler = self.get_sampler(
                loader.dataset,
                self.config["task"].get(
                    "eval_batch_size", self.config["task"]["batch_size"]
                ),
                shuffle=False,
                drop_last=True
            )

            loader = self.get_dataloader(
                loader.dataset,
                test_sampler,
            )

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        for task in self.normalizers:
            if self.normalizers.get(task, None) is not None:
                self.normalizers[task].to(self.device)

        predictions = {"id": []}
        predictions.update({task: [] for task in self.task_names})
        predictions.update({key: [] for key in other_output_keys_in_prediction})

        run_predict(loader)

        if isinstance(loader, torch.utils.data.dataloader.DataLoader) and \
                (len(loader.batch_sampler) * loader.batch_sampler.batch_size * distutils.get_world_size() < len(loader.dataset)) and \
                distutils.is_master():
            aux_val_dataset = Subset(loader.dataset,
                                     range(len(loader.batch_sampler) * loader.batch_sampler.batch_size * distutils.get_world_size(),
                                           len(loader.dataset)))
            aux_val_loader = self.get_dataloader(
                aux_val_dataset,
                sampler=None,
                batch_size=self.config["task"].get(
                    "eval_batch_size", self.config["task"]["batch_size"]
                )
            )
            run_predict(aux_val_loader)

        self.save_results_async(
            predictions,
            results_file,
            keys=self.task_names+other_output_keys_in_prediction,
            per_image=per_image
        )

    def save_results_async(
        self, predictions, results_file: Optional[str], keys, per_image=False
    ):
        if results_file is None:
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )

        if per_image:
            for i in range(len(predictions["id"])):
                results_file_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "{}.pkl".format(predictions["id"][i]),
                )
                # save to a pickle
                with open(results_file_path, 'wb') as f:
                    pickle.dump({key: predictions[key][i] for key in predictions}, f)


