from abc import abstractmethod

import torch
import torch.nn as nn

from transformers import T5EncoderModel, EsmModel

from ocpmodels.common.registry import registry


@registry.register_model("huggingface_embedding")
class HuggingfaceEmbedding(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        other_dense_fields=None,
        *args,
        **kargs,
    ) -> None:
        super(HuggingfaceEmbedding, self).__init__()
        self.pt_model = self._get_pretrained_model(
            pretrained_model_name_or_path,
            *args, **kargs
        )
        self.pt_model.eval()
        self.other_dense_fields = other_dense_fields

    @abstractmethod
    def _get_pretrained_model(self, pretrained_model_name_or_path, *args, **kargs):
        """Derived classes should implement this function."""

    @abstractmethod
    def _postprocess(self, embedding_repr, batch):
        """Derived classes should implement this function."""

    def _forward(self, batch):
        input_idx = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            embedding_repr = self.pt_model(input_idx, attention_mask=attention_mask)
        return embedding_repr

    def forward(self, batch):
        embedding_repr = self._forward(batch)
        results = self._postprocess(embedding_repr, batch)
        if self.other_dense_fields is not None:
            results = torch.concat(
                [results] + [batch[field] for field in self.other_dense_fields],
                dim=-1
            )
        return results

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@registry.register_model("huggingface_t5_embedding")
class HuggingfaceT5Embedding(HuggingfaceEmbedding):
    def _get_pretrained_model(self, pretrained_model_name_or_path, *args, **kargs):
        model = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path + "/pytorch_model.bin",
            config=pretrained_model_name_or_path + "/config.json",
            *args, **kargs
        )
        return model

    def _postprocess(self, embedding_repr, batch):
        attention_mask = batch['attention_mask']
        results = (embedding_repr.last_hidden_state * attention_mask.unsqueeze(-1)).sum(axis=1) / attention_mask.sum(
            axis=1).unsqueeze(-1)
        return results


@registry.register_model("huggingface_esm_embedding")
class HuggingfaceEsmEmbedding(HuggingfaceEmbedding):
    def _get_pretrained_model(self, pretrained_model_name_or_path, *args, **kargs):
        model = EsmModel.from_pretrained(
            pretrained_model_name_or_path,
            *args, **kargs
        )
        return model

    def _postprocess(self, embedding_repr, batch):
        results = embedding_repr.last_hidden_state[:, 0, :]
        return results


@registry.register_model("huggingface_task")
class HuggingfaceTask(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super(HuggingfaceTask, self).__init__()
        self.config = config
        embedding_config = config.get("embedding", None)
        task_config = config.get("task", None)
        if embedding_config is not None:
            self.pt_model = registry.get_model_class(
                embedding_config["name"]
            )(**embedding_config["attributes"])
        else:
            self.pt_model = None
        if task_config is not None:
            self.task = registry.get_model_class(
                task_config["name"]
            )(**task_config["attributes"])
        else:
            self.task = None

    def forward(self, batch):
        if self.pt_model is not None:
            batch = self.pt_model(batch)
        if self.task is not None:
            batch = self.task(batch)
        return batch

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
