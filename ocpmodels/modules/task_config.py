import torch.nn as nn
import torch.optim as optim
from ocpmodels.modules import scheduler
from ocpmodels.modules import normalizer
from ocpmodels.modules import torchmetrics_evaluator


def multitask_metrics(config, **kargs):
    return torchmetrics_evaluator.MultitaskMetrics(config, **kargs)


def multitask_meters(config, **kargs):
    return torchmetrics_evaluator.MultitaskMeters(config, **kargs)


def nn_loss(config):
    if isinstance(config, list) and (len(config) == 1):
        config = config[0]
    return getattr(nn, config["name"])(**config["attributes"])


def multitask_losses(config):
    losses = {task["name"]: nn_loss(task["attributes"]) for task in config}
    return losses


def multitask_normalizers(config, loader, device):
    normalizers = {}
    for task in config:
        task_config = config[task]
        if task_config.get("normalize_label", False):
            if task_config.get("target_mean", None) is not None:
                normalizers[task_config["name"]] = normalizer.Normalizer(
                    mean=task_config["target_mean"],
                    std=task_config["target_std"],
                    device=device,
                )
            else:
                normalizers[task_config["name"]] = normalizer.Normalizer(
                    tensor=loader.dataset.data[task_config["name"]][
                        loader.dataset.__indices__
                    ],
                    device=device,
                )
        else:
            normalizers[task_config["name"]] = None
    return normalizers


def get_optimizer(config, model):
    optimizer_cls = getattr(optim, config.get("name", "AdamW"))

    if config.get("weight_decay", 0) > 0:
        # Do not regularize bias etc.
        params_decay = []
        params_no_decay = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                decay = True
                for i in config["weight_decay_containing"]:
                    if i in name:
                        params_no_decay += [param]
                        decay = False
                        break
                if decay:
                    params_decay += [param]

        optimizer = optimizer_cls(
            [
                {"params": params_no_decay, "weight_decay": 0},
                {
                    "params": params_decay,
                    "weight_decay": config["weight_decay"],
                },
            ],
            lr=config["lr_initial"],
            **config.get("optimizer_params", {}),
        )
    else:
        optimizer = optimizer_cls(
            params=model.parameters(),
            lr=config["lr_initial"],
            **config.get("optimizer_params", {}),
        )
    return optimizer


def get_scheduler(optimizer, config):
    return scheduler.LRScheduler(optimizer, config)
