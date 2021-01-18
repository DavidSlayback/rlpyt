from rlpyt.envs.gym_isaac.isaacgym_env import IsaacGymEnv
from rlpyt.samplers.serial.isaac import IsaacSampler
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


if __name__ == "__main__":
    sampler_args = {
        'batch_T': 32,
        'batch_B': 512
    }
    iter_size = sampler_args['batch_T'] * sampler_args['batch_B']
    nsteps = 1e8
    transfer_iter = nsteps // iter_size
    run_ID = 0
    affinity = dict(cuda_idx=0, workers_cpus=list(range(6)), alternating=False)
    # task = 'Ant'
    task = 'Ant'
    test = IsaacGymEnv(task)
    sampler = IsaacSampler(test, batch_T=sampler_args['batch_T'])
    import torch
    # Configs from pytorch_ppo_ant.yaml
    model_kwargs={
        'hidden_sizes': [256,128,64],
        'hidden_nonlinearity': torch.nn.SELU,
        'normalize_observation': False
    }

    PPO_kwargs={
        'learning_rate': 3e-4,
        'clip_vf_loss': False,
        'entropy_loss_coeff': 0.,
        'discount': 0.99,
        'linear_lr_schedule': False,
        'epochs': 10,
        'clip_grad_norm': 2.,
        'minibatches': 2,
        'normalize_rewards': None,
        'value_loss_coeff': 2.
    }
    agent = MujocoFfAgent(model_kwargs=model_kwargs)
    algo = PPO(**PPO_kwargs)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e8,
        log_interval_steps=1e4,
        affinity=affinity,
        transfer=True,
        transfer_iter=transfer_iter,
        # log_traj_window=10
    )
    config = dict(task=task)
    name = "ppo_nt_nv_" + task
    log_dir = "example_2a"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()