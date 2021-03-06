
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo, OCAlgo, OptInfoOC
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, AgentInputsOC, AgentInputsOCRnn
from rlpyt.utils.tensor import valid_mean,select_at_indexes
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs


class A2OC(OCAlgo):
    """
    Advantage Actor Option-Critic algorithm (synchronous).  Trains the agent by
    taking one gradient step on each iteration of samples, with advantages
    computed by generalized advantage estimation.
    """

    NAME = "A2OC"
    def __init__(
            self,
            discount=0.99,
            learning_rate=1e-3,  # Main learning rate
            termination_lr=5e-7,  # Termination learning rate
            pi_omega_lr=0.,  # policy over options learning rate
            interest_lr=1e-3,  # Learning rate for interest function
            value_loss_coeff=0.5,
            termination_loss_coeff=1.,  # Coefficient for termination loss component
            entropy_loss_coeff=0.01,  # Entropy loss for low-level policy
            omega_entropy_loss_coeff=0.01,  # Entropy loss for policy over options
            delib_cost=0.,  # Cost for switching options. Subtracted from rewards after normalization...Also added to termination advantage
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_termination_advantage=False,  # Normalize termination advantage? Doesn't seem to be done
            normalize_rewards=None,  # Can be 'return' (OpenAI, no mean subtraction), 'reward' (same as obs normalization) or None
            rew_clip=(-10, 10),  # Additional clipping for reward (if normalizing reward)
            rew_min_var=1e-6  # Minimum variance in running mean for reward (if normalizing reward)
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-5)
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
        self._batch_size = self.batch_spec.size  # For logging.

    def optimize_agent(self, itr, samples):
        """
        Train the agent on input samples, by one gradient step.
        """
        if hasattr(self.agent, "update_obs_rms"):
            # NOTE: suboptimal--obs sent to device here and in agent(*inputs).
            self.agent.update_obs_rms(samples.env.observation)
        self.optimizer.zero_grad()
        loss, pi_loss, value_loss, beta_loss, pi_omega_loss, entropy, entropy_o = self.loss(samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfoOC(
            loss=loss.item(),
            pi_loss=pi_loss.item(),
            q_loss=value_loss.item(),
            beta_loss=beta_loss.item(),
            pi_omega_loss=pi_omega_loss.item(),
            gradNorm=grad_norm.clone().detach().item(),  # backwards compatible,
            entropy=entropy.item(),
            pi_omega_entropy=entropy_o.item()
        )
        self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()

        return opt_info

    def loss(self, samples):
        """
        Computes the training loss: policy_loss + value_loss + entropy_loss.
        Policy loss: log-likelihood of actions * advantages
        Value loss: 0.5 * (estimated_value - return) ^ 2
        Organizes agent inputs from training samples, calls the agent instance
        to run forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_inputs = AgentInputsOC(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
            sampled_option=samples.agent.agent_info.o,
        )
        if self.agent.recurrent:
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T = 0.
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            po = samples.agent.agent_info.prev_o
            (dist_info_o, q, beta, dist_info_omega), _rnn_state = self.agent(*agent_inputs, po, init_rnn_state, device=agent_inputs.prev_action.device)
        else:
            dist_info_o, q, beta, dist_info_omega = self.agent(*agent_inputs, device=agent_inputs.prev_action.device)
        dist = self.agent.distribution
        dist_omega = self.agent.distribution_omega
        # TODO: try to compute everyone on device.
        return_, advantage, valid, beta_adv, not_init_states, op_adv = self.process_returns(samples)

        logli = dist.log_likelihood(samples.agent.action, dist_info_o)
        pi_loss = - valid_mean(logli * advantage, valid)

        o = samples.agent.agent_info.o
        q_o = select_at_indexes(o, q)
        value_error = 0.5 * (q_o - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # Termination loss
        prev_o = samples.agent.agent_info.prev_o
        beta_prev_o = select_at_indexes(prev_o, beta)
        beta_error = beta_prev_o * beta_adv
        beta_loss = self.termination_loss_coeff * valid_mean(beta_error, not_init_states)

        logli = dist_omega.log_likelihood(o, dist_info_omega)
        # pi_omega_loss = - valid_mean(logli * advantage, valid)
        pi_omega_loss = - valid_mean(logli * op_adv, valid)

        entropy = dist.mean_entropy(dist_info_o, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy
        entropy_o = dist_omega.mean_entropy(dist_info_omega, valid)
        entropy_loss_omega = - self.omega_entropy_loss_coeff * entropy_o

        loss = pi_loss + pi_omega_loss + beta_loss + value_loss + entropy_loss + entropy_loss_omega

        # perplexity = dist.mean_perplexity(dist_info_o, valid)

        return loss, pi_loss, value_loss, beta_loss, pi_omega_loss, entropy, entropy_o
