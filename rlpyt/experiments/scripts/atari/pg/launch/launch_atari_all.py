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

runs_per_setting = 3  # 3 runs
# Paths
path_a2c = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "atari_ff_a2c_gpu.py").as_posix()
path_ppo = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "atari_ff_ppo_gpu.py").as_posix()
path_a2oc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "atari_ff_a2oc_gpu.py").as_posix()
path_ppoc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "atari_ff_ppoc_gpu.py").as_posix()
# Default keys
default_key = 'montezuma'
int_key = 'montezuma_interest'
extrawide_key = 'montezuma_4x'
# Param options
A2C_LRS = list(zip([3e-4, 1e-3, 3e-3, 1e-2]))
NOC_FC_SIZES = list(zip([512, 2048]))
PPO_LRS = list(zip([3e-4, 1e-3, 3e-3]))
OC_DELIB = list(zip([0., 0.02]))
games = list(zip(['montezuma_revenge', 'hero', 'breakout']))
# Variant keys
lr_key = [("algo", "learning_rate")]
delib_key = [("algo", "delib_cost")]
fc_key = [("model", "fc_sizes")]
interest_key = [("model", "use_interest")]
game_key = [("env", "game")]
# Common directory names
lr_names_a2c = ["LR_{}".format(*v) for v in A2C_LRS]
lr_names_ppo = ["LR_{}".format(*v) for v in PPO_LRS]
delib_names = ["D_{}".format(*v) for v in OC_DELIB]
game_names = ["{}".format(*v) for v in games]
fc_names = ["MFC_{}".format(*v) for v in NOC_FC_SIZES]



# A2C
experiment_title = "A2C_Atari"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
variant_levels.append(VariantLevel(fc_key, NOC_FC_SIZES, fc_names))  # Smaller or larger model
variant_levels.append(VariantLevel(lr_key, A2C_LRS, lr_names_a2c))  # Learning rates
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_a2c,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_key,),
)

# PPO
experiment_title = "PPO_Atari"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
variant_levels.append(VariantLevel(fc_key, NOC_FC_SIZES, fc_names))  # Smaller or larger model
variant_levels.append(VariantLevel(lr_key, PPO_LRS, lr_names_ppo))  # Learning rates
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_ppo,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_key,),
)

# A2OC
experiment_title = "A2OC_Atari"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Deliberation cost
variant_levels.append(VariantLevel(lr_key, A2C_LRS, lr_names_a2c))  # Learning rates
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_a2oc,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_key,),
)

# PPOC
experiment_title = "PPOC_Atari"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, games, game_names))  # Games
variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Deliberation cost
variant_levels.append(VariantLevel(lr_key, PPO_LRS, lr_names_ppo))  # Learning rates
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_ppoc,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_key,),
)