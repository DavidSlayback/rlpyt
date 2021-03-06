import copy
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=1e-3,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.98,
        linear_lr_schedule=True,
        minibatches=4,
        epochs=4,
    ),
    env=dict(game="pong"),
    model=dict(fc_sizes=512),
    optim=dict(),
    runner=dict(
        n_steps=1e7,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=64,
        batch_B=32,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config
config = copy.deepcopy(config)
config["env"]["game"] = "montezuma_revenge"
configs['montezuma'] = config
config = copy.deepcopy(config)
config["model"]["fc_sizes"] = 2048  # Match 4 options worth of parameters
configs['montezuma_4x'] = config
