import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.envs.gym_pomdps.gym_pomdp_env import OPTIMAL_RETURNS

# Takeaways from graphs
# Best RNN: 256 (vs 128, 64) unit gru (vs lstm)
# 2 and 4 options are equivalent
# 0 and 0.05 deliberation costs are equivalent
# 
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
# ***BEST RNN: 256 size gru***
# RNN = list(zip(['lstm']))
RNN_SIZE = list(zip([64, 128, 256]))
FOMDP = list(zip([False, True]))
INTEREST = list(zip([False, True]))  # Interest is better. 2 options is equivalent. Delib cost of 0-0.05
NUM_OPTIONS = list(zip([2, 4]))
OC_DELIB = list(zip([0., 0.05, 0.5]))
# ENVS = list(zip(list(OPTIMAL_RETURNS.keys())))  # All environments with comparable optimal returns
ENVS = list(zip(['POMDP-hallway-continuing-v0', 'POMDP-hallway2-continuing-v0', ]))# 'POMDP-rock_sample_5_6-continuing-v0']))  # Subselect
ENV = list(zip(['POMDP-rock_sample_5_4-continuing-v2']))  # Subselect
ENVS_PLUS_PARAMS = list(zip([30, 30], [100, 100], ['POMDP-hallway-continuing-v0', 'POMDP-hallway2-continuing-v0'], [int(5e5), int(2e6)]))  # B, T, ENVS, N_STEPS
ENV_PLUS_B_T = list(zip([6], [100], ['POMDP-rock_sample_5_4-continuing-v2']))
# Variant keys
lr_key = [("algo", "learning_rate")]
batch_B_key=[("sampler", "batch_B")]
batch_T_key=[("sampler", "batch_T")]
delib_key = [("algo", "delib_cost")]
fc_key = [("model", "fc_sizes")]
interest_key = [("model", "use_interest")]
nopt_key = [("model", "option_size")]
rnn_type_key = [("model", "rnn_type")]
rnn_size_key = [("model", "rnn_size")]
game_key = [("env", "id")]
nstep_key = [("runner", "n_steps")]
env_plus_B_key = batch_B_key + game_key
B_T_env_key = batch_B_key + batch_T_key + game_key
envs_plus_params_key = batch_B_key + batch_T_key + game_key + nstep_key
fomdp_key = [("env", "fomdp")]
# Common directory names
# env_names = ["{}".format(*v) for v in ENVS]
env_names = ["{}".format(*v) for v in ENV]
env_name = ["{}".format(*v) for v in ENV]
rnn_names = ["{}".format(*v) for v in RNN]
rnn_size_names = ["RNNH_{}".format(*v) for v in RNN_SIZE]
delib_names = ["DELIB_{}".format(*v) for v in OC_DELIB]
obs_names = ["OMNI_{}".format(*v) for v in FOMDP]
int_names = ["INT_{}".format(*v) for v in INTEREST]
nopt_names = ["NOPT_{}".format(*v) for v in NUM_OPTIONS]

# A2C
experiment_title = "A2C_Pomdp"
variant_levels = list()
variant_levels.append(VariantLevel(B_T_env_key, ENV_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
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

# A2C RNN
experiment_title = "A2CRnn_Pomdp"
variant_levels = list()
variant_levels.append(VariantLevel(B_T_env_key, ENV_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
variant_levels.append(VariantLevel(rnn_size_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_a2c_rnn,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_key_rnn,),
)
#
# A2OC
experiment_title = "A2OC_Pomdp"
variant_levels = list()
variant_levels.append(VariantLevel(B_T_env_key, ENV_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Option deliberation cost
variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
variants, log_dirs = make_variants(*variant_levels)
run_experiments(
    script=path_a2oc,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(oc_key,),
)

# A2OC RNN
experiment_title = "A2OCRnn_Pomdp"
variant_levels = list()
variant_levels.append(VariantLevel(B_T_env_key, ENV_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Option deliberation cost
variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
variant_levels.append(VariantLevel(rnn_size_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
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
