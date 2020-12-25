
from rlpyt.envs.wrappers import RLPYT_MINIGRID_WRAPPERS, RLPYT_WRAPPER_KEY
from rlpyt.models.params import CONVNET_MINIGRID_TINY, CONVNET_MINIGRID_ENSEMBLES
configs = dict()
env_args = dict()
env_args[RLPYT_WRAPPER_KEY] = RLPYT_MINIGRID_WRAPPERS


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=1e-3,
        value_loss_coeff=.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=False,
    ),
    env=dict(id="MiniGrid-FourRooms-v0", **env_args),
    model=dict(scale_obs=True, obs_mean=4., obs_scale=8., **CONVNET_MINIGRID_TINY),
    optim=dict(),
    runner=dict(
        seed=None,
        n_steps=10e6,
        log_interval_steps=1e3,
        log_traj_window=10
    ),
    sampler=dict(
        batch_T=64,
        batch_B=32,
        max_decorrelation_steps=100,
    ),
)

configs["BaseMiniGrid"] = config
