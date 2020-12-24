
import copy
from rlpyt.envs.wrappers import ClipActionsWrapper, RLPYT_WRAPPER_KEY

configs = dict()
env_args = dict()
env_args[RLPYT_WRAPPER_KEY] = [ClipActionsWrapper]

config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-5,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        value_loss_coeff=0.5,
        normalize_advantage=True,
    ),
    env=dict(id="Hopper-v3", **env_args),
    model=dict(normalize_observation=False),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=2e4,
    ),
    sampler=dict(
        batch_T=100,
        batch_B=8,
        max_decorrelation_steps=1000,
    ),
)
configs["a2c_1M"] = config
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_grad_norm=0.5,
        entropy_loss_coeff=0.0,
        value_loss_coeff=0.5,
        normalize_advantage=True,
        normalize_rewards='return',
        rew_clip=(-10, 10),
        rew_min_var=(1e-6)
    ),
    env=dict(id="HalfCheetah-Directional-v0", **env_args),
    model=dict(normalize_observation=True, baselines_init=True),
    optim=dict(),
    runner=dict(
        seed=None,
        n_steps=1e6,
        log_interval_steps=1e3,
        transfer=True,
        transfer_iter=150,
        log_traj_window=10
    ),
    sampler=dict(
        batch_T=256,
        batch_B=8,
        max_decorrelation_steps=100,
    ),
)
configs["a2c_1M_halfcheetahtransfer"] = config


