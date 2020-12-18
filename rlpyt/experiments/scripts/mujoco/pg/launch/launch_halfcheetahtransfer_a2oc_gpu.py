import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_a2oc_gpu.py"

script = path.as_posix()
# script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ff_a2c_gpu.py"
affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=6,
    n_socket=1,
    alternating=True
)
runs_per_setting = 6
default_config_key = "a2oc_1M_halfcheetahtransfer"
experiment_title = "A2OC_Transfer"
variant_levels = list()

lrs = [3e-5, 1e-4, 3e-4]
values = list(zip(lrs))
dir_names = ["A2OC_lr_{}".format(*v) for v in values]
keys = [("algo", "learning_rate")]
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
