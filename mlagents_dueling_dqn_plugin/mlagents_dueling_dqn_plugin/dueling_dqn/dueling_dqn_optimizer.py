from typing import Any, Dict, List, Optional, Tuple, Union, cast

import attr
import numpy as np
from mlagents.torch_utils import torch, nn, default_device
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents_envs.timers import timed
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import (
    NetworkSettings,
    OffPolicyHyperparamSettings,
    ScheduleType,
    TrainerSettings,
)
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.decoders import ValueHeads
from mlagents.trainers.torch_entities.networks import Actor, Critic, NetworkBody
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


@attr.s(auto_attribs=True)
class DuelingDQNSettings(OffPolicyHyperparamSettings):
    gamma: float = 0.99
    exploration_schedule: ScheduleType = ScheduleType.LINEAR
    exploration_initial_eps: float = 0.1
    exploration_final_eps: float = 0.05
    exploration_steps: int = 20000
    target_update_interval: int = 10000
    tau: float = 0.005
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update


class DuelingDQNOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)

        self.hyperparameters: DuelingDQNSettings = cast(
            DuelingDQNSettings, trainer_settings.hyperparameters
        )

        params = list(self.policy.actor.parameters())
        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )

        self.stream_names = list(self.reward_signals.keys())
        self.gammas = [_val.gamma for _val in trainer_settings.reward_signals.values()]
        self.use_dones_in_backup = {
            name: int(not self.reward_signals[name].ignore_done)
            for name in self.stream_names
        }
        self.tau = self.hyperparameters.tau

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_exploration_rate = ModelUtils.DecayedValue(
            self.hyperparameters.exploration_schedule,
            self.hyperparameters.exploration_initial_eps,
            self.hyperparameters.exploration_final_eps,
            self.hyperparameters.exploration_steps,
        )

        self.q_net_target = DuelingQNetwork(
            stream_names=self.reward_signals.keys(),
            observation_specs=policy.behavior_spec.observation_specs,
            network_settings=policy.network_settings,
            action_spec=policy.behavior_spec.action_spec,
        )
        ModelUtils.soft_update(self.policy.actor, self.q_net_target, 1.0)
        self.q_net_target.to(default_device())

    @property
    def critic(self):
        return self.q_net_target

    @staticmethod
    def _gather_flat_q_for_actions(
        q_values: torch.Tensor, actions: torch.Tensor, action_spec: ActionSpec
    ) -> torch.Tensor:
        if len(action_spec.discrete_branches) == 1:
            return torch.gather(q_values, dim=1, index=actions.long())

        selected = []
        start = 0
        for branch_i, branch_size in enumerate(action_spec.discrete_branches):
            branch_q = q_values[:, start : start + branch_size]
            branch_action = actions[:, branch_i : branch_i + 1].long()
            selected.append(torch.gather(branch_q, dim=1, index=branch_action))
            start += branch_size
        return torch.cat(selected, dim=1)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        exp_rate = self.decay_exploration_rate.get_value(self.policy.get_current_step())
        self.policy.actor.exploration_rate = exp_rate

        rewards: Dict[str, torch.Tensor] = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer(batch, n_obs)]
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in ObsUtil.from_buffer_next(batch, n_obs)]
        actions = AgentAction.from_buffer(batch)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])

        current_q_values, _ = self.policy.actor.critic_pass(
            current_obs, sequence_length=self.policy.sequence_length
        )

        with torch.no_grad():
            next_q_values_list, _ = self.q_net_target.critic_pass(
                next_obs, sequence_length=self.policy.sequence_length
            )
            greedy_actions = self.policy.actor.get_greedy_action(next_q_values_list)

        qloss = []
        action_spec = self.policy.behavior_spec.action_spec
        for name_i, name in enumerate(rewards.keys()):
            with torch.no_grad():
                next_selected = self._gather_flat_q_for_actions(
                    next_q_values_list[name], greedy_actions, action_spec
                )
                next_q_values = next_selected.mean(dim=1)
                target_q_values = rewards[name] + (
                    (1.0 - self.use_dones_in_backup[name] * dones)
                    * self.gammas[name_i]
                    * next_q_values
                )
                target_q_values = target_q_values.reshape(-1, 1)

            curr_q = self._gather_flat_q_for_actions(
                current_q_values[name], actions.discrete_tensor, action_spec
            )
            target_for_loss = target_q_values.expand_as(curr_q)
            qloss.append(torch.nn.functional.smooth_l1_loss(curr_q, target_for_loss))

        loss = torch.mean(torch.stack(qloss))
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ModelUtils.soft_update(self.policy.actor, self.q_net_target, self.tau)

        update_stats = {
            "Losses/Value Loss": loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/epsilon": exp_rate,
        }
        for reward_provider in self.reward_signals.values():
            update_stats.update(reward_provider.update(batch))
        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self.critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules


class DuelingQNetwork(nn.Module, Actor, Critic):
    MODEL_EXPORT_VERSION = 3

    def __init__(
        self,
        stream_names: List[str],
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        exploration_initial_eps: float = 1.0,
    ):
        if action_spec.continuous_size > 0:
            raise UnityTrainerException("Dueling DQN supports only discrete actions.")
        if action_spec.discrete_size <= 0:
            raise UnityTrainerException("Dueling DQN requires at least one discrete action branch.")

        nn.Module.__init__(self)
        self.exploration_rate = exploration_initial_eps
        self.action_spec = action_spec
        self.action_size = int(sum(action_spec.discrete_branches))

        self.network_body = NetworkBody(observation_specs, network_settings)
        if network_settings.memory is not None:
            encoding_size = network_settings.memory.memory_size // 2
        else:
            encoding_size = network_settings.hidden_units

        self.value_heads = ValueHeads(list(stream_names), encoding_size, 1)
        self.advantage_heads = ValueHeads(list(stream_names), encoding_size, self.action_size)

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.action_spec.continuous_size)]), requires_grad=False
        )
        self.discrete_act_size_vector = torch.nn.Parameter(
            torch.Tensor([self.action_spec.discrete_branches]), requires_grad=False
        )
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([int(self.network_body.memory_size)]), requires_grad=False
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size

    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def critic_pass(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.forward_q(inputs, memories=memories, sequence_length=sequence_length)

    def forward_q(
        self,
        inputs: List[torch.Tensor],
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        encoding, memories = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )
        values = self.value_heads(encoding)
        advantages = self.advantage_heads(encoding)

        q_outputs: Dict[str, torch.Tensor] = {}
        for name, adv in advantages.items():
            val = values[name].unsqueeze(-1)
            q_outputs[name] = val + adv - adv.mean(dim=1, keepdim=True)
        return q_outputs, memories

    def _mask_flat_q(self, flat_q: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
        if masks is None:
            return flat_q
        return flat_q * masks - 1e8 * (1.0 - masks)

    def _split_branches(self, flat_q: torch.Tensor) -> List[torch.Tensor]:
        branches = []
        start = 0
        for branch_size in self.action_spec.discrete_branches:
            branches.append(flat_q[:, start : start + branch_size])
            start += branch_size
        return branches

    def get_random_action(
        self, batch_size: int, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        actions = []
        start = 0
        device = default_device()
        for branch_size in self.action_spec.discrete_branches:
            if masks is None:
                action = torch.randint(0, branch_size, (batch_size, 1), device=device)
            else:
                branch_mask = masks[:, start : start + branch_size]
                probs = branch_mask / torch.clamp(branch_mask.sum(dim=1, keepdim=True), min=1.0)
                action = torch.multinomial(probs, 1)
            actions.append(action)
            start += branch_size
        return torch.cat(actions, dim=1)

    def get_greedy_action(
        self, q_values: Dict[str, torch.Tensor], masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        all_q = torch.cat([val.unsqueeze(0) for val in q_values.values()])
        flat_q = all_q.sum(dim=0)
        flat_q = self._mask_flat_q(flat_q, masks)
        branch_actions = [torch.argmax(branch_q, dim=1, keepdim=True) for branch_q in self._split_branches(flat_q)]
        return torch.cat(branch_actions, dim=1)

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        out_vals, memories_out = self.critic_pass(inputs, memories, sequence_length)
        batch_size = len(inputs[0])
        disc_action_out = self.get_greedy_action(out_vals, masks)
        deterministic_disc_action_out = disc_action_out
        export_out = [self.version_number, self.memory_size_vector]
        export_out += [
            disc_action_out,
            self.discrete_act_size_vector,
            deterministic_disc_action_out,
        ]
        if self.network_body.memory_size > 0:
            export_out += [memories_out]
        return tuple(export_out)

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        deterministic: bool = False,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        run_out: Dict[str, Any] = {}
        batch_size = len(inputs[0])
        if not deterministic and np.random.rand() < self.exploration_rate:
            action_tensor = self.get_random_action(batch_size, masks)
        else:
            out_vals, memories = self.critic_pass(inputs, memories, sequence_length)
            action_tensor = self.get_greedy_action(out_vals, masks)

        discrete_actions = [
            action_tensor[:, branch_i : branch_i + 1]
            for branch_i in range(action_tensor.shape[1])
        ]
        action_out = AgentAction(None, discrete_actions)
        run_out["env_action"] = action_out.to_action_tuple()
        return action_out, run_out, memories if memories is not None else torch.Tensor([])
