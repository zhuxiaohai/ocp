import numpy as np
import torch

from ocpmodels.modules.torchmetrics_evaluator import (
    MultitaskMeters,
    collapse_dict,
)


def test_multitask_meters():
    config = [
        {'name': 't1', 'attributes': [{"rename": "m1", "name": "MeanMetric", "attributes": {}}]},
        {'name': 't2', 'attributes': [{"rename": "m2", "name": "MeanMetric", "attributes": {}}]},
    ]
    meter = MultitaskMeters(config)
    prediction = {
        "t1": torch.tensor([2, 1, 2, 0, 1, 2, 2, 2]),
        "t2": torch.tensor([1, 2, 3]),
    }
    meter.update(prediction)
    result = meter.compute()
    assert "t1_m1" in result
    assert "t2_m2" in result
    assert result["t1_m1"] == 1.5000
    assert result["t2_m2"] == 2.
    prediction = {
        "t1": torch.tensor([5, 6, 7]),
        "t2": 10,
    }
    result = meter(prediction)
    assert "t1" in result
    assert "m1" in result["t1"]
    assert result["t1"]["m1"] == 6.0
    assert "t2" in result
    assert "m2" in result["t2"]
    assert result["t2"]["m2"] == 10.0
    result = meter.compute()
    assert torch.round(result["t1_m1"], decimals=4) == 2.7273
    assert torch.round(result["t2_m2"], decimals=4) == 4.0
    meter.reset()
    prediction = {
        "t1": torch.tensor([2, 1, 2, 0, 1, 2, 2, 2]),
        "t2": torch.tensor([1, 2, 3]),
    }
    result = meter(prediction)
    assert torch.round(result["t1"]["m1"], decimals=4) == 1.5
    assert torch.round(result["t2"]["m2"], decimals=4) == 2.0
    result = meter.compute()
    assert torch.round(result["t1_m1"], decimals=4) == 1.5000
    assert torch.round(result["t2_m2"], decimals=4) == 2.


def test_collapse_dict():
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    assert collapse_dict(d) == {'a': 1, 'b_c': 2, 'b_d_e': 3}
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': {'g': 4}}
    assert collapse_dict(d) == {'a': 1, 'b_c': 2, 'b_d_e': 3, 'f_g': 4}
    d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': {'g': 4, 'h': {'i': 5}}}
    assert collapse_dict(d) == {'a': 1, 'b_c': 2, 'b_d_e': 3, 'f_g': 4, 'f_h_i': 5}
