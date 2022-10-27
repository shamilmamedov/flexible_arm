"""Different kind of nets that can be used."""
from abc import abstractmethod
from typing import Sequence
import math
import numpy as np

import torch
import torch.nn as nn
import gym.spaces as spaces


def _create_mlp(dims: Sequence[int], relu_at_end: bool = False):
    """Creates a simple relu multi layer perceptron."""
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(dim_in, dim_out))

        if relu_at_end or idx < len(dims) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim_out))

    return nn.Sequential(*layers)


class ParamMPCNet(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        param_space: spaces.Box,
        action_shape: Sequence[int],
    ) -> None:
        super().__init__()
        assert len(param_space.shape) == 1

        self.observation_shape = observation_shape
        self.observation_dim = np.prod(observation_shape)
        self.param_dim = param_space.shape[0]
        self.param_space = param_space
        self.action_shape = action_shape
        self.action_dim = np.prod(action_shape)

    def forward(self, observation: torch.Tensor, mpc_param: torch.Tensor):
        observation = observation.flatten(start_dim=1)  # Keep batch dim.
        return self._forward(observation, mpc_param).reshape((-1, *self.action_shape))

    @abstractmethod
    def _forward(self, observation: torch.Tensor, mpc_param: torch.Tensor):
        ...


class ObservationNet(ParamMPCNet):
    """The param is just part of the input."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        param_space: spaces.Box,
        action_shape: Sequence[int],
        hidden_dims: Sequence[int] = (64,),
    ):
        super().__init__(observation_shape, param_space, action_shape)
        input_dim = self.observation_dim + self.param_dim
        dims = [input_dim, *hidden_dims, self.action_dim]

        self.mlp = _create_mlp(dims)

    def _forward(self, observation: torch.Tensor, mpc_param: torch.Tensor):
        inputs = torch.concat([observation, mpc_param], dim=1)

        return self.mlp(inputs)


class EmbeddingNet(ParamMPCNet):
    """We embed the parameters."""

    prefix = "embed"

    def __init__(
        self,
        observation_shape: Sequence[int],
        param_space: spaces.Box,
        action_shape: Sequence[int],
        hidden_dims_before_embed: Sequence[int] = tuple(),
        hidden_dims_after_embed: Sequence[int] = (64,),
        encoding_dim: int = 64,
    ):
        super().__init__(observation_shape, param_space, action_shape)

        dims_before_embed = [self.observation_dim, *hidden_dims_before_embed]
        dims_after_embed = [
            dims_before_embed[-1],
            *hidden_dims_after_embed,
            self.action_dim,
        ]

        self.mlp_before_embed = _create_mlp(dims_before_embed, relu_at_end=True)
        self.mlp_after_embed = _create_mlp(dims_after_embed)

        self.encoding_dim = encoding_dim
        embed_layers = [
            nn.Linear(encoding_dim * self.param_dim, dims_before_embed[-1]),
            nn.ReLU(),
        ]
        self.embed_layer = nn.Sequential(*embed_layers)

    def cosine_embed(self, mpc_param: torch.Tensor):
        low = torch.tensor(self.param_space.low, device=mpc_param.device)
        high = torch.tensor(self.param_space.high, device=mpc_param.device)

        normalized_param = (mpc_param - low) / (high - low)

        indices = torch.arange(0, self.encoding_dim, device=mpc_param.device)

        embed_param = torch.einsum("bp,e->bep", normalized_param, indices)
        embed_param = torch.cos(math.pi * embed_param)

        return embed_param.reshape(-1, self.param_dim * self.encoding_dim)

    def _forward(self, observation: torch.Tensor, mpc_param: torch.Tensor):
        inputs = self.mlp_before_embed(observation)

        cosine_param = self.cosine_embed(mpc_param)
        embed_param = self.embed_layer(cosine_param)

        return self.mlp_after_embed(inputs * embed_param)


class EnsembleNet(ParamMPCNet):
    def __init__(
        self,
        observation_dim: int,
        param_space: spaces.Box,
        action_shape: Sequence[int],
        ensemble_nets: Sequence[ParamMPCNet],
    ) -> None:
        super().__init__(observation_dim, param_space, action_shape)
        self.ensemble_nets = ensemble_nets

    def forward(self, observation: torch.Tensor, mpc_param: torch.Tensor):
        if self.traning:
            return []
