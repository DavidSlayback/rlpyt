import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=1,
    n_socket=1,
    alternating=True
)

runs_per_setting = 3  # 3 runs
# Paths
path_a2c = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "procgen_ff_a2c_gpu.py").as_posix()
path_ppo = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "procgen_ff_ppo_gpu.py").as_posix()
path_a2oc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "procgen_ff_a2oc_gpu.py").as_posix()
path_ppoc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "procgen_ff_ppoc_gpu.py").as_posix()
# Default keys
default_key = 'base'
oc_key = 'base_4_oc'
int_key = 'base_interest'
# Param options
A2C_LRS = list(zip([3e-4, 1e-3, 3e-3, 1e-2]))
NOC_FC_SIZES = list(zip([512, 2048]))
PPO_LRS = list(zip([3e-4, 1e-3, 3e-3]))
OC_DELIB = list(zip([0., 0.02]))
games = list(zip(['fruitbot', 'coinrun', 'caveflyer']))
# Variant keys
lr_key = [("algo", "learning_rate")]
delib_key = [("algo", "delib_cost")]
fc_key = [("model", "fc_sizes")]
interest_key = [("model", "use_interest")]
game_key = [("env", "game")]

game_names = ["{}".format(*v) for v in games]
# PPO
# experiment_title = "PPO_Procgen"
# variant_levels = list()
# variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
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

# PPOC
experiment_title = "PPOC_Procgen"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_ppoc,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(oc_key,),
)