import torchmetrics
import torch


def collection_metrics(config, **kargs):
    if config[0].get("rename", None) is not None:
        metrics = torchmetrics.MetricCollection(
            {metric["rename"]: getattr(torchmetrics, metric["name"])(**metric["attributes"], **kargs) for metric in config}
        )
    else:
        metrics = torchmetrics.MetricCollection(
            [getattr(torchmetrics, metric["name"])(**metric["attributes"], **kargs) for metric in config]
        )
    return metrics


def collapse_dict(d, prefix=''):
    """
    Recursively collapse a nested dictionary by concatenating the keys of the first and second layers
    and all the way down to the last layer until the final dict is a one-dimensional dict.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result.update(collapse_dict(v, prefix=prefix+k+'_'))
        else:
            result[prefix+k] = v
    return result


def convert_nested_dict_tensor(nested_dict):
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            convert_nested_dict_tensor(v)
        elif isinstance(v, torch.Tensor):
            nested_dict[k] = v.item()
        else:
            pass
    return nested_dict


def combine_dict(dict1, dict2):
    for key in dict2:
        if key in dict1:
            combine_dict(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1


class MultitaskMetrics(torchmetrics.MultitaskWrapper):
    """
    MultitaskWrapper with a compute method that returns a flattened dictionary of metrics.
    """
    def __init__(self, config, **kargs):
        super(MultitaskMetrics, self).__init__(
            {task["name"]: collection_metrics(task["attributes"], **kargs) for task in config}
        )
        self.config = config
        if hasattr(self, "sync_on_compute") and (kargs.get("sync_on_compute", None) is not None):
            self.sync_on_compute = kargs["sync_on_compute"]

    def compute(self):
        metrics = super(MultitaskMetrics, self).compute()
        if len(metrics) > 1:
            return collapse_dict(metrics)
        else:
            key = list(metrics.keys())[0]
            return collapse_dict(metrics[key])


class MultitaskMeters:
    def __init__(self, config, **kargs):
        self.config = config
        self.meters = {
            task["name"]: collection_metrics(task["attributes"], **kargs) for task in config
        }

    def to(self, device):
        for key in self.meters:
            self.meters[key] = self.meters[key].to(device)
        return self

    def compute(self):
        metrics = {key: self.meters[key].compute() for key in self.meters}
        if len(metrics) > 1:
            return collapse_dict(metrics)
        else:
            key = list(metrics.keys())[0]
            return collapse_dict(metrics[key])

    def update(self, Dict, weight=None):
        if weight is not None:
            for key in Dict:
                self.meters[key].update(Dict[key], weight[key])
        else:
            for key in Dict:
                self.meters[key].update(Dict[key])

    def reset(self):
        for key in self.meters:
            self.meters[key].reset()

    def __call__(self, Dict, weight=None):
        if weight is not None:
            metrics = {key: self.meters[key](Dict[key], weight[key]) for key in Dict}
        else:
            metrics = {key: self.meters[key](Dict[key]) for key in Dict}
        return metrics
