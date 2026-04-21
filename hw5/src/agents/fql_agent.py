from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        noise = torch.randn((1, self.action_dim), device=observation.device)
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        action = self.onestep_actor(observation, noise)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        dt = 1.0 / self.flow_steps
        action = noise 
        batch_size = observation.shape[0]

        for i in range(self.flow_steps):
            t = torch.full(
                (batch_size, 1),
                i / self.flow_steps,
                device = observation.device,
                dtype = observation.dtype,
            )
            action = action + dt * self.bc_actor(observation, action, t)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1)

        with torch.no_grad():
            noise = torch.randn(
                (next_observations.shape[0], self.action_dim),
                device=next_observations.device,
                dtype=next_observations.dtype,
            )
            next_actions = self.onestep_actor(next_observations, noise)
            next_actions = torch.clamp(next_actions, -1, 1)

            next_q = self.target_critic(next_observations, next_actions)
            next_v = torch.mean(next_q, dim=0)  # average of the two target Qs
            target_q = rewards + self.discount * (1.0 - dones) * next_v

        q = self.critic(observations, actions)
        loss = ((q - target_q.unsqueeze(0)) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        batch_size = observations.shape[0]
        noise = torch.randn(
            (batch_size, self.action_dim),
            device=observations.device,
            dtype=observations.dtype,
        )
        t = torch.rand(
            (batch_size, 1),
            device=observations.device,
            dtype=observations.dtype,
        )
        a_tilde = (1.0 - t) * noise + t * actions
        target = actions - noise

        pred = self.bc_actor(observations, a_tilde, t)
        loss = ((pred - target) ** 2).mean(dim=-1).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        batch_size = observations.shape[0]
        noise = torch.randn(
            (batch_size, self.action_dim),
            device=observations.device,
            dtype = observations.dtype,
        )
        onestep_actions = self.onestep_actor(observations, noise)
        with torch.no_grad():
            bc_actions = self.get_bc_action(observations, noise)
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        distill_loss = self.alpha * ((onestep_actions - bc_actions) ** 2).mean(dim = -1).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        clipped_onestep_actions = torch.clamp(onestep_actions, -1, 1)
        q_values = self.critic(observations, clipped_onestep_actions)
        q_loss = -torch.mean(q_values, dim=0).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = ((clipped_onestep_actions - actions) ** 2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        for target_param, critic_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.mul_(1.0 - self.target_update_rate)
            target_param.data.add_(self.target_update_rate * critic_param.data)
