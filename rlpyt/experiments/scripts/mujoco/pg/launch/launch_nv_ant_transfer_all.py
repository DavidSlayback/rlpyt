import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

PHYSX_N_THREADS = 4  # By default, isaac environments use 4 cores by themselves. Each sampler will use its own as well
affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=2,  # This thing chews GPU memory
    n_socket=1,
    alternating=False
)

runs_per_setting = 3  # 3 runs
# Paths
path_ppo = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "isaac_ff_ppo_gpu.py").as_posix()
path_ppoc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "isaac_ff_ppoc_gpu.py").as_posix()
# Default keys
default_key = 'nv_ant'
default_oc_key = 'nv_ant_oc'
# Param options
PPO_LRS = list(zip([1e-4, 3e-4, 1e-3]))
OC_DELIB = list(zip([0., 0.01, 1.]))
OC_SIZES = list(zip([2,4]))
tasks = list(zip(['Ant']))
# Variant keys
lr_key = [("algo", "learning_rate")]
delib_key = [("algo", "delib_cost")]
oc_size_key = [("model", "option_size")]
interest_key = [("model", "use_interest")]
game_key = [("env", "task")]


task_names = ["{}".format(*v) for v in tasks]

# experiment_title = "PPO_Isaac"
# variant_levels = list()
# variant_levels.append(VariantLevel(game_key, tasks, task_names))  # Games
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_ppo,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_key,),
# )

experiment_title = "PPOC_Isaac"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, tasks, task_names))  # Games
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_ppoc,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_oc_key,),
)