
import copy
from rlpyt.envs.wrappers import ClipActionsWrapper, RLPYT_WRAPPER_KEY

configs = dict()
env_args = dict()
env_args[RLPYT_WRAPPER_KEY] = [ClipActionsWrapper]
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=True,
        # bootstrap_timelimit=False,
    ),
    env=dict(id="Hopper-v3", **env_args),
    model=dict(normalize_observation=False),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=2048 * 10,
    ),
    sampler=dict(
        batch_T=2048,
        batch_B=1,
        max_decorrelation_steps=0,
    ),
)

configs["ppo_1M_serial"] = config

config = copy.deepcopy(configs["ppo_1M_serial"])
config["sampler"]["batch_B"] = 8
config["sampler"]["batch_T"] = 256
configs["ppo_1M_cpu"] = config
config = copy.deepcopy(configs["ppo_1M_serial"])
config["algo"]["minibatches"] = 1
config["algo"]["epochs"] = 32
configs["ppo_32ep_1mb"] = config
config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=0.5,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=False,
        # bootstrap_timelimit=False,
        clip_vf_loss=False,
        normalize_rewards='return',
        rew_clip=(-10,10),
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

configs["ppo_1M_halfcheetahtransfer"] = config
config = copy.deepcopy(configs["ppo_1M_halfcheetahtransfer"])
config["env"] = dict(id='TMaze-TwoGoal-v0', **env_args)
config["algo"]["normalize_rewards"] = None
config["runner"]["log_traj_window"] = 20
configs["ppo_1M_TMaze"] = config
