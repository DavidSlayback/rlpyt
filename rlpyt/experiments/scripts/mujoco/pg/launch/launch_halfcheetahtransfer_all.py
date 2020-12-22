import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=6,
    n_socket=1,
    alternating=True
)

runs_per_setting = 6  # 6 runs

# A2C
path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_a2c_gpu.py"
script = path.as_posix()
default_config_key = "a2c_1M_halfcheetahtransfer"
experiment_title = "A2C_HalfCheetahDirectionalTransfer"
variant_levels = list()
lrs = [3e-4, 1e-3, 5e-3, 1e-2]
values = list(zip(lrs))
dir_names = ["A2C_lr_{}".format(*v) for v in values]
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

# PPO
# path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_ppo_gpu.py"
# script = path.as_posix()
# default_config_key = "ppo_1M_halfcheetahtransfer"
# experiment_title = "PPO_HalfCheetahDirectionalTransfer"
# variant_levels = list()
# lrs = [3e-5, 1e-4, 3e-4]
# values = list(zip(lrs))
# dir_names = ["PPO_lr_{}".format(*v) for v in values]
# keys = [("algo", "learning_rate")]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )


# # PPOC
# path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_ppoc_gpu.py"
# script = path.as_posix()
# default_config_key = "ppoc_1M_halfcheetahtransfer"
# experiment_title = "PPOC_HalfCheetahDirectionalTransfer"
# variant_levels = list()
# lrs = [3e-5, 1e-4, 3e-4]
# delib=[0., 1e-1, 1.]
# values = list(zip(delib))
# dir_names = ["PPOC_{}".format(*v) for v in values]
# keys = [("algo", "delib_cost")]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# values = list(zip(lrs))
# keys = [("algo", "learning_rate")]
# dir_names = ["lr_{}".format(*v) for v in values]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )
#
# # A2OC
# path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_a2oc_gpu.py"
# script = path.as_posix()
# default_config_key = "a2oc_1M_halfcheetahtransfer"
# experiment_title = "A2OC_HalfCheetahDirectionalTransfer"
# variant_levels = list()
# lrs = [3e-5, 1e-4, 3e-4]
# delib=[0., 1e-1, 1.]
# values = list(zip(delib))
# dir_names = ["A2OC_{}".format(*v) for v in values]
# keys = [("algo", "delib_cost")]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# values = list(zip(lrs))
# keys = [("algo", "learning_rate")]
# dir_names = ["lr_{}".format(*v) for v in values]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# variants, log_dirs = make_variants(*variant_levels)
#
# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )

# PPIOC
path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_ppoc_gpu.py"
script = path.as_posix()
default_config_key = "ppioc_1M_halfcheetahtransfer"
experiment_title = "PPIOC_HalfCheetahDirectionalTransfer"
variant_levels = list()
# lrs = [3e-5, 1e-4, 3e-4]
# delib=[0., 1e-1, 1.]
lrs = [3e-4, 1e-3, 5e-3]
delib=[0., 1e-1, 1.]
values = list(zip(delib))
dir_names = ["PPIOC_{}".format(*v) for v in values]
keys = [("algo", "delib_cost")]
variant_levels.append(VariantLevel(keys, values, dir_names))
values = list(zip(lrs))
keys = [("algo", "learning_rate")]
dir_names = ["lr_{}".format(*v) for v in values]
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

# # A2IOC
# path = pathlib.Path(__file__).resolve().parent.parent / 'train' / "mujoco_ff_a2oc_gpu.py"
# script = path.as_posix()
# default_config_key = "a2ioc_1M_halfcheetahtransfer"
# experiment_title = "A2OC_HalfCheetahDirectionalTransfer"
# variant_levels = list()
# lrs = [3e-5, 1e-4, 3e-4]
# delib=[0., 1e-1, 1.]
# values = list(zip(delib))
# dir_names = ["A2IOC_{}".format(*v) for v in values]
# keys = [("algo", "delib_cost")]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# values = list(zip(lrs))
# keys = [("algo", "learning_rate")]
# dir_names = ["lr_{}".format(*v) for v in values]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# variants, log_dirs = make_variants(*variant_levels)
#
# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )