from copy import deepcopy

configs = dict()

# Base configuration from the paper's code repository
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.999,
        learning_rate=5e-4,
        gae_lambda=0.95,
        epochs=3,
        clip_vf_loss=True,
        ratio_clip=0.2,
        minibatches=8,
        entropy_loss_coeff=0.01,
        clip_grad_norm=0.5,
        value_loss_coeff=0.5,
        normalize_advantage=True,
        normalize_rewards='return',  # Typical Baselines reward normalization
        rew_clip=(-10., 10.),
        rew_min_var=1e-8
    ),
    env=dict(
        game='coinrun',
        start_level=0,
        num_levels=500,
        distribution_mode='easy'
    ),
    model=dict(),  # They use IMPALA model with residual blocks, [16, 32, 32], 256 final fc size. No observation normalization
    optim=dict(),
    runner=dict(
        seed=None,  # Seed is irrelevant anyway, start_level is equivalent
        n_steps=25e6,
        log_interval_steps=1e5
    ),
    sampler=dict(
        batch_T=256,
        batch_B=64,
        max_decorrelation_steps=100,
    ),
)
configs['base'] = config
config = deepcopy(config)
config['env']['distribution_mode'] = 'hard'
config['runner']['n_steps'] = 2e8
config['sampler']['batch_B'] = 256  # They actually use 4 separate learning processes, so this is rough
configs['base_hard'] = config