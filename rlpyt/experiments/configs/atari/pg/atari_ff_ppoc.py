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
        delib_cost=0.
    ),
    env=dict(game="montezuma_revenge"),
    model=dict(option_size=4, use_interest=False, fc_sizes=512),
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

configs["montezuma"] = config
config = copy.deepcopy(config)
config["model"]["use_interest"] = True
configs['montezuma_interest'] = config