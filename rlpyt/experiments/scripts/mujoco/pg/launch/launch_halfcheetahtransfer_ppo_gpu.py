
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
import pathlib

path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_ppo_gpu.py"

script = path.as_posix()
# script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ff_ppo_gpu.py"
affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=6,
    n_socket=1,
    alternating=True
)
runs_per_setting = 1
default_config_key = "ppo_1M_halfcheetahtransfer"
experiment_title = "PPO_Transfer"
variant_levels = list()

seeds = [1,2,3,4,5,6]
values = list(zip(seeds))
dir_names = ["PPO_seed_{}".format(*v) for v in values]
keys = [("runner", "seed")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
