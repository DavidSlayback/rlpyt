"""Double actor-critic models

Double actor-critic models use OptionGaussianActorCriticModel
Option-critic uses same, as does PPOC


So models are the same...


DAC_A2C:
    compute_pi_hat(): Prob of options
    compute_pi_bar(o, a, mean, std): Log prob of action
    compute_log_pi_a:
        If mdp is hat: log_prob of selected options
        If bar: log_prob of selected actions (under options)
    compute_adv: Get v, adv, ret for mdp. Compute GAE. DEtach
    learn: Compute pi and v loss for mdp
    step():
        Get prediction
        compute pi_hat (pio), sample option
        Sample a, compute_pi_bar for logprob
        v_bar = q[sampled_o]
        v_hat = (q[sampled_o] * pi_hat).sum
        compute loss for both pi/v_hat and pi/v_bar
DAC_PPO is similar, but with PPO clipping for both pi_hat and pi_bar

IOPG:
    Each worker
        pre_step: Compute pi_hat, sample option
        Sample action, compute log_prob
"""