from copy import deepcopy

# Common args
# batch_T = 100  # Updated to match episode size from Chris's advice
batch_T = 20  # On empirical evaluation, smaller seems to do alright and runs faster. 10 and 20 were rougly equivalent
batch_B = 30  # Reduce for larger POMDPs
nsteps = 5e5  # Is fine for hallway1, bump for hallway2 and rocksample

base_sampler_args = dict(batch_T=batch_T, batch_B=batch_B, max_decorrelation_steps=0)
base_runner_args = dict(n_steps=nsteps, log_interval_steps=1e3, seed=None)
base_env_args = dict(fomdp=False, id='POMDP-hallway-continuing-v0', time_limit=100)  # Partially-observable, time_limit as in cassandra's thesis
base_oc_model_args = dict(option_size=4, use_interest=False, use_diversity=False, use_attention=False)  # OC model args
base_oc_algo_args = dict(termination_lr=0., pi_omega_lr=0., interest_lr=0., delib_cost=0.)
base_rnn_args = dict(rnn_type='gru', rnn_size=256, rnn_placement=1, layer_norm=True)  # Base rnn args, best I've seen


configs = dict()
config = dict(
    agent=dict(),
    algo=dict(
        # No discount here, use environment's
        learning_rate=1e-3,
        clip_grad_norm=2.,
        entropy_loss_coeff=0.01,
        value_loss_coeff=1.,
        gae_lambda=0.95,
        normalize_advantage=True,
        linear_lr_schedule=False
    ),
    env=dict(**base_env_args),
    model=dict(
        hidden_sizes=[64, 64],
        shared_processor=True
    ),
    optim=dict(),
    sampler=dict(**base_sampler_args),
    runner=dict(**base_runner_args)
)
configs['hallway_5e5'] = config
config = deepcopy(config)
config['model'] = {**config['model'], **base_oc_model_args}
config['algo'] = {**config['algo'], **base_oc_algo_args}
configs['hallway_5e5_oc'] = config
config = deepcopy(configs['hallway_5e5'])
config['model'] = {**config['model'], **base_rnn_args}
configs['hallway_5e5_rnn'] = config
config = deepcopy(configs['hallway_5e5_oc'])
config['model'] = {**config['model'], **base_rnn_args}
configs['hallway_5e5_rnn_oc'] = config
