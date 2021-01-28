
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo, OCAlgo, OptInfoOC
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, AgentInputsOC, AgentInputsOCRnn
from rlpyt.utils.tensor import valid_mean,select_at_indexes
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "option", "prev_option", "return_", "advantage", "op_adv", "termination_advantage",
     "valid", "not_init_states", "old_dist_info_o", "old_dist_info_omega", "old_q"])


class PPOC(OCAlgo):
    """
    Proximal Policy Option Critic algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the low-level policy loss.
    """

    NAME = "PPOC"
    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            termination_lr=0.001,  # Termination learning rate
            pi_omega_lr=0.001,  # policy over options learning rate
            interest_lr=0.001,  # Learning rate for interest function
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
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_termination_advantage=False,  # Normalize termination advantage? Doesn't seem to be done
            clip_vf_loss=False,  # Clip VF_loss as in OpenAI?
            clip_pi_omega_loss=False,  # Clip policy over option loss in a similar way to PPO? Not done by others
            clip_beta_loss=False,
            normalize_rewards=None,  # Can be 'return' (OpenAI, no mean subtraction), 'reward' (same as obs normalization) or None
            rew_clip=(-10, 10),  # Additional clipping for reward (if normalizing reward)
            rew_min_var=1e-6  # Minimum variance in running mean for reward (if normalizing reward)
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-5)
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputsOC(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
            sampled_option=samples.agent.agent_info.o,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid, beta_adv, not_init_states, op_adv = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            option=samples.agent.agent_info.o,
            prev_option=samples.agent.agent_info.prev_o,
            return_=return_,
            advantage=advantage,
            op_adv=op_adv,
            termination_advantage=beta_adv,
            valid=valid,
            not_init_states=not_init_states,
            old_dist_info_o=samples.agent.agent_info.dist_info_o,
            old_dist_info_omega=samples.agent.agent_info.dist_info_omega,
            old_q=samples.agent.agent_info.q
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfoOC(*([] for _ in range(len(OptInfoOC._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, value_loss, beta_loss, pi_omega_loss, entropy, entropy_o = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                # OptInfoOC = namedtuple("OptInfo", ["loss", "pi_loss", "q_loss", "beta_loss", "pi_omega_loss", "gradNorm", "entropy", "pi_omega_entropy"])
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.q_loss.append(value_loss.item())
                opt_info.beta_loss.append(beta_loss.item())
                opt_info.pi_omega_loss.append(pi_omega_loss.item())
                opt_info.gradNorm.append(grad_norm.clone().detach().item())  # backwards compatible
                opt_info.entropy.append(entropy.item())
                opt_info.pi_omega_entropy.append(entropy_o.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, o, prev_o, return_, advantage, op_adv, beta_adv, valid, not_init_states, old_dist_info_o,
             old_dist_info_omega, old_q, init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            (dist_info_o, q, beta, dist_info_omega), _rnn_state = self.agent(*agent_inputs, init_rnn_state, device=action.device)
        else:
            dist_info_o, q, beta, dist_info_omega = self.agent(*agent_inputs, device=action.device)
        dist = self.agent.distribution
        dist_omega = self.agent.distribution_omega

        # Surrogate policy loss
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info_o,
            new_dist_info=dist_info_o)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        # Surrogate value loss (if doing)
        q_o = select_at_indexes(o, q)
        old_q_o = select_at_indexes(o, old_q)
        if self.clip_vf_loss:
            v_loss_unclipped = (q_o - return_) ** 2
            v_clipped = old_q_o + torch.clamp(q_o - old_q_o, -self.ratio_clip, self.ratio_clip)
            v_loss_clipped = (v_clipped - return_) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            value_error = 0.5 * v_loss_max.mean()
        else:
            value_error = 0.5 * (q_o - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # Termination loss
        beta_prev_o = select_at_indexes(prev_o, beta)
        beta_error = beta_prev_o * beta_adv
        beta_loss = self.termination_loss_coeff * valid_mean(beta_error, not_init_states)

        # Pi_omega loss. Surrogate (if using)
        if self.clip_pi_omega_loss:
            ratio = dist_omega.likelihood_ratio(o, old_dist_info=old_dist_info_omega,
            new_dist_info=dist_info_omega)
            # surr_1 = ratio * advantage
            surr_1 = ratio * op_adv
            clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
                                        1. + self.ratio_clip)
            # surr_2 = clipped_ratio * advantage
            surr_2 = clipped_ratio * op_adv
            surrogate = torch.min(surr_1, surr_2)
            pi_omega_loss = - valid_mean(surrogate, valid)
        else:
            logli = dist_omega.log_likelihood(o, dist_info_omega)
            # pi_omega_loss = - valid_mean(logli * advantage, valid)
            pi_omega_loss = - valid_mean(logli * op_adv, valid)

        entropy = dist.mean_entropy(dist_info_o, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy
        entropy_o = dist_omega.mean_entropy(dist_info_omega, valid)
        entropy_loss_omega = - self.omega_entropy_loss_coeff * entropy_o

        loss = pi_loss + pi_omega_loss + value_loss + beta_loss + entropy_loss + entropy_loss_omega

        # perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, pi_loss, value_loss, beta_loss, pi_omega_loss, entropy, entropy_o
