"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim 
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h 
        
        layers.append(nn.Linear(input_dim, action_dim * chunk_size))
        self.network = nn.Sequential(*layers)
        self.mse = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0] 
        pred = self.network(state)
        pred = pred.view(B, self.chunk_size, self.action_dim)
        loss = self.mse(pred, action_chunk) 
        return loss 


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        pred = self.network(state)
        pred = pred.view(B, self.chunk_size, self.action_dim)
        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        in_dim = state_dim + chunk_size * action_dim + 1
        out_dim = chunk_size * action_dim 

        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, out_dim))

        self.network = nn.Sequential(*layers)
        self.mse = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        # a0 ~ N(0, I)
        a0 = torch.randn_like(action_chunk)
        # tau = N(0, 1)
        tau = torch.rand(B, 1, device = state.device, dtype = state.dtype)
        # interpeloate 
        tau_b = tau.view(B, 1, 1)
        a_tau = tau_b * action_chunk + (1 - tau_b) * a0
        # A_t - A_0
        target_velocity = action_chunk - a0

        a_tau_flat = a_tau.view(B, self.chunk_size * self.action_dim)
        network_input = torch.cat([state, a_tau_flat, tau], dim = 1)

        pred = self.network(network_input)
        pred = pred.view(B, self.chunk_size, self.action_dim)
        loss = self.mse(pred, target_velocity)
        return loss 



    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        a = torch.randn(B, self.chunk_size, self.action_dim, device=state.device, dtype=state.dtype)

        dt = 1.0 / num_steps 
        for step in range(num_steps):
            tau_val = step / num_steps 
            tau = torch.full((B, 1), tau_val, device=state.device, dtype=state.dtype)
            a_flat = a.view(B, self.chunk_size * self.action_dim)
            network_input = torch.cat([state, a_flat, tau], dim = 1)
            velocity = self.network(network_input)
            velocity = velocity.view(B, self.chunk_size, self.action_dim)
            a = a + velocity * dt
        return a

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
