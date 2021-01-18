import copy

nv_batch_B_ant = 512  # num_envs for ant
nv_batch_T_base = 32
nv_batch_T_rlg = 16

batch_B = nv_batch_B_ant
batch_T = nv_batch_T_base
nsteps = 1e8
niters = nsteps // (batch_B*batch_T)
transfer_iter = int(niters // 2)
configs = dict()
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=2.,
        value_loss_coeff=2.,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=2,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=False,
        clip_vf_loss=False,
        normalize_rewards=None
    ),
    env=dict(task='Ant'),
    model=dict(
        normalize_observation=False,
        hidden_sizes=[256, 128, 64],
        hidden_nonlinearity='SELU'  # Convert after making environment
    ),
    optim=dict(),
    runner=dict(
        n_steps=1e8,
        log_interval_steps=1e4,
        transfer=True,
        transfer_iter=transfer_iter
    ),
    sampler=dict(
        batch_T=32,
        batch_B=512,
        max_decorrelation_steps=100,
    ),
)
configs['nv_ant'] = config
option_model_kwargs = {
    'option_size': 4,
    'use_interest': False
}
option_algo_kwargs = {
    'termination_loss_coeff': 1.,
    'delib_cost': 0.,
    'omega_entropy_loss_coeff': 0.01,
    'normalize_termination_advantage': False,
    'clip_pi_omega_loss': False,
    'clip_beta_loss': False
}
config = copy.deepcopy(config)
config['model'].update(option_model_kwargs)
config['algo'].update(option_algo_kwargs)
configs['nv_ant_oc'] = config