import torch
import torch.nn as nn
from ocpmodels.common.registry import registry


@registry.register_model("mlp_block")
class MLPBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(MLPBlock, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        x = self.tanh(x)
        return x

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@registry.register_model("mlp")
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_labels, dropout_rate):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.n_blocks = len(hidden_sizes)
        self.blocks = nn.ModuleList()
        self.out_proj = nn.Linear(hidden_sizes[-1], num_labels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_labels = num_labels

        for i in range(self.n_blocks):
            block = MLPBlock(input_size, self.hidden_sizes[i], dropout_rate)
            self.blocks.append(block)
            input_size = hidden_sizes[i]

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@registry.register_model("concat_dense")
class ConcatDense(nn.Module):
    def __init__(
        self,
        fields,
    ) -> None:
        super(ConcatDense, self).__init__()
        self.fields = fields

    def forward(self, batch):
        # concatenete all the fields of batch, to get a single dense tensor
        output = torch.concat([batch[field] for field in self.fields], dim=1)
        return output

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())