import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.envs.gym_pomdps.gym_pomdp_env import OPTIMAL_RETURNS

affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=6,
    n_socket=1,
    alternating=False
)

runs_per_setting = 6  # 3 runs
# Paths
path_a2c = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "pomdp_ff_a2c_gpu.py").as_posix()
path_a2oc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "pomdp_ff_a2oc_gpu.py").as_posix()
path_a2c_rnn = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "pomdp_rnn_a2c_gpu.py").as_posix()
path_a2oc_rnn = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "pomdp_rnn_a2oc_gpu.py").as_posix()
# Default keys
default_key = 'hallway_5e5'
oc_key = 'hallway_5e5_oc'
default_key_rnn = 'hallway_5e5_rnn'
oc_key_rnn = 'hallway_5e5_rnn_oc'
# Param options
RNN = list(zip(['lstm', 'gru']))
RNN = list(zip(['lstm']))
RNN_SIZE = list(zip([64, 128, 256]))
FOMDP = list(zip([False, True]))
INTEREST = list(zip([False, True]))
NUM_OPTIONS = list(zip([2, 4]))
ENVS = list(zip(list(OPTIMAL_RETURNS.keys())))  # All environments with comparable optimal returns
# Variant keys
lr_key = [("algo", "learning_rate")]
delib_key = [("algo", "delib_cost")]
fc_key = [("model", "fc_sizes")]
interest_key = [("model", "use_interest")]
nopt_key = [("model", "option_size")]
rnn_type_key = [("model", "rnn_type")]
rnn_size_key = [("model", "rnn_size")]
game_key = [("env", "id")]
fomdp_key = [("env", "fomdp")]
# Common directory names
env_names = ["{}".format(*v) for v in ENVS]
rnn_names = ["{}".format(*v) for v in RNN]
rnn_size_names = ["RNNH_{}".format(*v) for v in RNN_SIZE]
obs_names = ["OMNI_{}".format(*v) for v in FOMDP]
int_names = ["INT_{}".format(*v) for v in INTEREST]
nopt_names = ["NOPT_{}".format(*v) for v in NUM_OPTIONS]

# A2C
# experiment_title = "A2C_Pomdp"
# variant_levels = list()
# variant_levels.append(VariantLevel(game_key, ENVS, env_names))  # pomdps
# variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_a2c,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_key,),
# )

# A2C RNN
# experiment_title = "A2CRnn_Pomdp"
# variant_levels = list()
# variant_levels.append(VariantLevel(game_key, ENVS, env_names))  # pomdps
# variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
# variant_levels.append(VariantLevel(rnn_type_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_a2c_rnn,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_key_rnn,),
# )

# A2OC
# experiment_title = "A2OC_Pomdp"
# variant_levels = list()
# variant_levels.append(VariantLevel(game_key, ENVS, env_names))  # pomdps
# variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
# variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_a2oc,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(oc_key,),
# )

# A2OC RNN
experiment_title = "A2OCRnn_Pomdp"
variant_levels = list()
variant_levels.append(VariantLevel(game_key, ENVS, env_names))  # pomdps
variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
variant_levels.append(VariantLevel(rnn_type_key, INTEREST, int_names))  # Use of interest function
variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
variant_levels.append(VariantLevel(rnn_type_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_a2oc_rnn,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(oc_key_rnn,),
)
