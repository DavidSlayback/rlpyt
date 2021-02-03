# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple actor-critic agent implemented in JAX + Haiku."""

from typing import Any, Callable, NamedTuple, Tuple

from bsuite.baselines import base
import dm_env
from dm_env import specs
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
import numpy as np

class OCTrajectory(NamedTuple):
    """A trajectory is a sequence of observations, actions, rewards, discounts.

    Note: `observations` should be of length T+1 to make up the final transition.
    """
    observations: np.ndarray  # [T + 1, ...]
    actions: np.ndarray  # [T]
    prev_o: np.ndarray  # [T]
    o: np.ndarray  # [T]
    rewards: np.ndarray  # [T]
    discounts: np.ndarray  # [T]

class OCBuffer:
    """A simple buffer for accumulating trajectories."""

    _observations: np.ndarray
    _actions: np.ndarray
    _prev_o: np.ndarray
    _o: np.ndarray
    _rewards: np.ndarray
    _discounts: np.ndarray

    _max_sequence_length: int
    _needs_reset: bool = True
    _t: int = 0

    def __init__(
            self,
            obs_spec: specs.Array,
            action_spec: specs.Array,
            max_sequence_length: int,
    ):
        """Pre-allocates buffers of numpy arrays to hold the sequences."""
        self._observations = np.zeros(
            shape=(max_sequence_length + 1, *obs_spec.shape), dtype=obs_spec.dtype)
        self._actions = np.zeros(
            shape=(max_sequence_length, *action_spec.shape),
            dtype=action_spec.dtype)
        self._prev_o = np.zeros(max_sequence_length, dtype=action_spec.dtype)
        self._o = np.zeros(max_sequence_length, dtype=action_spec.dtype)
        self._rewards = np.zeros(max_sequence_length, dtype=np.float32)
        self._discounts = np.zeros(max_sequence_length, dtype=np.float32)

        self._max_sequence_length = max_sequence_length

    def append(
            self,
            timestep: dm_env.TimeStep,
            action: base.Action,
            new_timestep: dm_env.TimeStep,
    ):
        """Appends an observation, action, reward, and discount to the buffer."""
        if self.full():
            raise ValueError('Cannot append; sequence buffer is full.')

        # Start a new sequence with an initial observation, if required.
        if self._needs_reset:
            self._t = 0
            self._observations[self._t] = timestep.observation
            self._needs_reset = False

        # Append (o, a, r, d) to the sequence buffer.
        self._observations[self._t + 1] = new_timestep.observation
        self._actions[self._t] = action
        self._rewards[self._t] = new_timestep.reward
        self._discounts[self._t] = new_timestep.discount
        self._t += 1

        # Don't accumulate sequences that cross episode boundaries.
        # It is up to the caller to drain the buffer in this case.
        if new_timestep.last():
            self._needs_reset = True

    def append_options(self,
                       prev_o: base.Action,
                       o: base.Action,
                       ):
        if self.full():
            raise ValueError('Cannot append; sequence buffer is full.')
        self._prev_o[self._t] = prev_o
        self._o[self._t] = o

    def drain(self) -> OCTrajectory:
        """Empties the buffer and returns the (possibly partial) trajectory."""
        if self.empty():
            raise ValueError('Cannot drain; sequence buffer is empty.')
        trajectory = OCTrajectory(
            self._observations[:self._t + 1],
            self._actions[:self._t],
            self._prev_o[:self._t],
            self._o[:self._t],
            self._rewards[:self._t],
            self._discounts[:self._t],
        )
        self._t = 0  # Mark sequences as consumed.
        self._needs_reset = True
        return trajectory

    def empty(self) -> bool:
        """Returns whether or not the trajectory buffer is empty."""
        return self._t == 0

    def full(self) -> bool:
        """Returns whether or not the trajectory buffer is full."""
        return self._t == self._max_sequence_length

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax
import rlax

Logits = jnp.ndarray
Value = jnp.ndarray
LSTMState = Any
Logits_Omega = jnp.ndarray
Beta = jnp.ndarray
Q = jnp.ndarray
PolicyValueNet = Callable[[jnp.ndarray], Tuple[Logits, Value]]
OptionCriticNet = Callable[[jnp.ndarray], Tuple[Logits, Beta, Q, Logits_Omega]]


class TrainingState(NamedTuple):
  params: hk.Params
  o: jnp.ndarray  # Selected option
  prev_o: jnp.ndarray  # Previous option
  opt_state: Any


class OptionCritic(base.Agent):
  """Feed-forward actor-critic agent."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      n_options: int,
      network: OptionCriticNet,
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      sequence_length: int,
      discount: float,
      td_lambda: float,
  ):

    # Define loss function.
    def loss(trajectory: OCTrajectory) -> jnp.ndarray:
      """"Actor-critic loss."""
      logits, betas, qs, pi_omegas = network(trajectory.observations)
      logits, values = network(trajectory.observations)
      td_errors = rlax.td_lambda(
          v_tm1=values[:-1],
          r_t=trajectory.rewards,
          discount_t=trajectory.discounts * discount,
          v_t=values[1:],
          lambda_=jnp.array(td_lambda),
      )
      critic_loss = jnp.mean(td_errors**2)
      actor_loss = rlax.policy_gradient_loss(
          logits_t=logits[:-1],
          a_t=trajectory.actions,
          adv_t=td_errors,
          w_t=jnp.ones_like(td_errors))

      return actor_loss + critic_loss

    # Transform the loss into a pure function.
    loss_fn = hk.without_apply_rng(hk.transform(loss, apply_rng=True)).apply

    # Define update function.
    @jax.jit
    def sgd_step(state: TrainingState,
                 trajectory: sequence.Trajectory) -> TrainingState:
      """Does a step of SGD over a trajectory."""
      gradients = jax.grad(loss_fn)(state.params, trajectory)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)
      return TrainingState(params=new_params, opt_state=new_opt_state)

    # Initialize network parameters and optimiser state.
    init, forward = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_observation = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)
    initial_params = init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)
    initial_option_state = jax.random.randint(next(rng), (1, ), 0, n_options)

    # Internalize state.
    self._state = TrainingState(initial_params, initial_option_state, initial_option_state, initial_opt_state)
    self._forward = jax.jit(forward)
    self._buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
    self._sgd_step = sgd_step
    self._rng = rng

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions according to a softmax policy."""
    key = next(self._rng)
    observation = timestep.observation[None, ...]
    logits, beta, q, pi_omega = self._forward(self._state.params, observation)
    prev_o = self._state.prev_o
    
    # logits, _ = self._forward(self._state.params, observation)
    action = jax.random.categorical(key, logits).squeeze()
    return int(action)

  def _select_option(self, pi_omega: jnp.ndarray) -> base.Action:
      """Selects options according to a softmax policy"""
      key = next(self._rng)
      option = jax.random.categorical(key, pi_omega).squeeze()
      return int(option)

  def _sample_termination(self, beta: jnp.ndarray) -> bool:
      """Determines termination from termination probabilities"""
      key = next(self._rng)
      termination = jax.random.bernoulli(key, beta)
      return bool(termination)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds a transition to the trajectory buffer and periodically does SGD."""
    self._buffer.append(timestep, action, new_timestep)
    if self._buffer.full() or new_timestep.last():
      trajectory = self._buffer.drain()
      self._state = self._sgd_step(self._state, trajectory)


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray,
                  n_options: int,
                  use_interest: bool = True,
                  seed: int = 0) -> base.Agent:
  """Creates an option-critic agent with default hyperparameters."""

  def network(inputs: jnp.ndarray) -> Tuple[Logits, Beta, Q, Logits_Omega]:
    flat_inputs = hk.Flatten()(inputs)  # Inputs flattened
    torso = hk.nets.MLP([64, 64])  # Shared state processor, 2x64 with relu after each.
    # Option outputs
    policy_over_options_head = hk.Sequential(hk.Linear(n_options), Partial(jax.nn.softmax, axis=-1))
    beta_head = hk.Sequential(hk.Linear(n_options), jax.nn.sigmoid)
    interest_head = hk.Sequential(hk.Linear(n_options), jax.nn.sigmoid)
    q_head = hk.Linear(n_options)
    # q_ent_head = hk.Linear(n_options)
    policy_head = hk.Sequential(hk.Linear(action_spec.num_values * n_options),
                                Partial(jnp.reshape, newshape=(-1, n_options, action_spec.num_values)),
                                Partial(jax.nn.softmax, axis=-1)
                                )
    embedding = torso(flat_inputs)
    logits = policy_head(embedding)
    beta = beta_head(embedding)
    interest = interest_head(embedding)
    pi_omega = policy_over_options_head(embedding)
    pi_omega = pi_omega * interest
    pi_omega = pi_omega / jnp.sum(pi_omega, axis=-1)  # Normalized interest policy
    # q_ent = q_ent_head(embedding)
    q = q_head(embedding)
    return logits, beta, q, pi_omega

  def network_no_interest(inputs: jnp.ndarray) -> Tuple[Logits, Beta, Q, Logits_Omega]:
    flat_inputs = hk.Flatten()(inputs)  # Inputs flattened
    torso = hk.nets.MLP([64, 64])  # Shared state processor, 2x64 with relu after each.
    # Option outputs
    policy_over_options_head = hk.Sequential(hk.Linear(n_options), Partial(jax.nn.softmax, axis=-1))
    beta_head = hk.Sequential(hk.Linear(n_options), jax.nn.sigmoid)
    q_head = hk.Linear(n_options)
    # q_ent_head = hk.Linear(n_options)
    policy_head = hk.Sequential(hk.Linear(action_spec.num_values * n_options),
                                Partial(jnp.reshape, newshape=(-1, n_options, action_spec.num_values)),
                                Partial(jax.nn.softmax, axis=-1)
                                )
    embedding = torso(flat_inputs)
    logits = policy_head(embedding)
    beta = beta_head(embedding)
    pi_omega = policy_over_options_head(embedding)
    # q_ent = q_ent_head(embedding)
    q = q_head(embedding)
    return logits, beta, q, pi_omega

  return OptionCritic(
      obs_spec=obs_spec,
      action_spec=action_spec,
      n_options=n_options,
      network=network if use_interest else network_no_interest,
      optimizer=optax.adam(3e-3),
      rng=hk.PRNGSequence(seed),
      sequence_length=32,
      discount=0.99,
      td_lambda=0.9,
  )
