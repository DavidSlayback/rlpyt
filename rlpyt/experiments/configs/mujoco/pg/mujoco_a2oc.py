
import copy

configs = dict()

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
    env=dict(id="Hopper-v3"),
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
configs["a2oc_1M"] = config
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_grad_norm=1.,
        entropy_loss_coeff=0.0,
        omega_entropy_loss_coeff=0.01,
        value_loss_coeff=0.5,
        term_loss_coeff=1.,
        delib_cost=0.,
        normalize_advantage=True,
        normalize_rewards='return',
        rew_clip=(-10, 10),
        rew_min_var=(1e-6)
    ),
    env=dict(id="HalfCheetah-Directional-v0"),
    model=dict(normalize_observation=True, baselines_init=True, option_size=2),
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
configs["a2oc_1M_halfcheetahtransfer"] = config


