from copy import deepcopy

# Common args
batch_T = 20
batch_B = 30
nsteps = 5e5

base_sampler_args = dict(batch_T=batch_T, batch_B=batch_B, max_decorrelation_steps=50)
base_runner_args = dict(n_steps=nsteps, log_interval_steps=1e3, seed=None)
base_env_args = dict(fomdp=False, id='POMDP-hallway-continuing-v0', time_limit=100)  # Partially-observable, time_limit as in cassandra's thesis
base_oc_args = dict(option_size=4, use_interest=False, use_diversity=False, use_attention=False)  # OC model args
base_rnn_args = dict(rnn_type='lstm', rnn_size=128)  # Base rnn args


configs = dict()
config = dict(
    agent=dict(),
    algo=dict(
        # No discount here, use environment's
        learning_rate=1e-3,
        clip_grad_norm=2.,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.,
        normalize_advantage=True
    ),
    env=dict(**base_env_args),
    model=dict(
        hidden_sizes=[64, 64],
    ),
    optim=dict(),
    sampler=dict(**base_sampler_args),
    runner=dict(**base_runner_args)
)
configs['hallway_5e5'] = config
config = deepcopy(config)
config['model'] = {**config['model'], **base_oc_args}
configs['hallway_5e5_oc'] = config
config = deepcopy(configs['hallway_5e5'])
config['model'] = {**config['model'], **base_rnn_args}
configs['hallway_5e5_rnn'] = config
config = deepcopy(configs['hallway_5e5_oc'])
config['model'] = {**config['model'], **base_rnn_args}
configs['hallway_5e5_rnn_oc'] = config
