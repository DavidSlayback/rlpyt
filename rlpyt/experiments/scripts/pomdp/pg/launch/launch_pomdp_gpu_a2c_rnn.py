import pathlib
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.envs.gym_pomdps.gym_pomdp_env import OPTIMAL_RETURNS
import numpy as np

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

runs_per_setting = 3  # 3 runs
# Paths
path_a2c = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "b_pomdp_ff_a2c_gpu.py").as_posix()
path_a2oc = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "b_pomdp_ff_a2oc_gpu.py").as_posix()
path_a2c_rnn = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "b_pomdp_rnn_a2c_gpu.py").as_posix()
path_a2oc_rnn = (pathlib.Path(__file__).resolve().parent.parent / 'train' / "b_pomdp_rnn_a2oc_gpu.py").as_posix()
# Default keys
default_key = 'hallway_5e5'
oc_key = 'hallway_5e5_oc'
default_key_rnn = 'hallway_5e5_rnn'
oc_key_rnn = 'hallway_5e5_rnn_oc'
# Param options
# ***BEST RNN: 256 size gru***
# RNN = list(zip(['lstm']))
RNN_SIZE = list(zip([256, 512]))
rnn_size_key = [("model", "rnn_size")]
RNN_PLACE = list(zip([0, 1]))
rnn_place_key = [("model", "rnn_placement")]
SHARED_PROC = list(zip([False, True]))
shared_proc_key = [("model", "shared_processor")]
LAYER_NORM = list(zip([False, True]))
layer_norm_key = [("model", "layer_norm")]
lrs = list(zip([3e-4, 1e-3, 3e-3]))
lr_key = [("algo", "learning_rate")]
tlrs = list(zip([0., 5e-7]))
tlr_key = [("algo", "termination_lr")]
piolr_key = [("algo", "pi_omega_lr")]
intlr_key = [("algo", "interest_lr")]
delib_key = [("algo", "delib_cost")]

pas = list(zip(['None', 'Pi', 'V', 'PiV']))
pa_names = ["NoAct", "ActPi", "ActV", "ActPiV"]
pa_key = [("model", "prev_action")]
pas_s = list(zip(['None', 'PiV']))
pa_names_s = ["NoAct", "ActPiV"]

prs = list(zip(['None', 'Pi', 'V', 'PiV']))
pr_names = ["NoRew", "RewPi", "RewV", "RewPiV"]
prs_s = list(zip(['None', 'PiV']))
pr_names_s = ["NoRew", "RewPiV"]
pr_key = [("model", "prev_reward")]

po_key = [("model", "prev_option")]

fc_key = [("model", "fc_sizes")]
interest_key = [("model", "use_interest")]
FOMDP = list(zip([False, True]))
INTEREST = list(zip([False, True]))  # Interest is better. 2 options is equivalent. Delib cost of 0-0.05
NUM_OPTIONS = list(zip([2, 4]))
OC_DELIB = list(zip([0., 0.05, 0.5]))

# ENVS = list(zip(['POMDP-hallway-episodic-v0', 'POMDP-hallway2-episodic-v0']))# 'POMDP-rock_sample_5_6-continuing-v0']))  # Subselect

ENV = list(zip(['POMDP-rock_sample_5_4-continuing-v2']))  # Subselect
# ENVS_PLUS_PARAMS = list(zip([30, 30], [100, 100], ['POMDP-hallway-continuing-v0', 'POMDP-hallway2-continuing-v0'], [int(5e5), int(2e6)]))  # B, T, ENVS, N_STEPS
# ENVS_PLUS_PARAMS = list(zip([30, 30], [20, 20], ['POMDP-hallway-continuing-v0', 'POMDP-hallway2-continuing-v0'], [int(2e6), int(6e6)]))  # B, T, ENVS, N_STEPS
ENVS_PLUS_PARAMS = list(zip(['POMDP-shopping_5-continuing-v1', 'POMDP-heavenhell_4-continuing-v0', 'POMDP-rock_sample_5_4-continuing-v2', 'POMDP-tiger-continuing-v0'], [int(1e7), int(1e7), int(1e7), int(2e6)]))  # ENVS, N_STEPS
B_T = list(zip([10, 20, 50, 100]))
ENV_PLUS_B_T = list(zip([6], [100], ['POMDP-rock_sample_5_4-continuing-v2']))
# Variant keys
batch_B_key=[("sampler", "batch_B")]
batch_T_key=[("sampler", "batch_T")]
game_key = [("env", "id")]
nstep_key = [("runner", "n_steps")]
env_plus_B_key = batch_B_key + game_key
B_T_env_key = batch_B_key + batch_T_key + game_key
envs_plus_params_key = game_key + nstep_key
envs_plus_step_key = game_key + nstep_key
# ENVS_PLUS_STEPS = list(zip(['POMDP-hallway-episodic-v0', 'POMDP-hallway2-episodic-v0'], [int(2e6), int(5e6)]))
ENVS = list(zip(['POMDP-shopping_5-continuing-v1', 'POMDP-heavenhell_4-continuing-v0', 'POMDP-rock_sample_5_4-continuing-v2', 'POMDP-tiger-continuing-v0']))
ENVS_PLUS_STEPS = list(zip(['POMDP-hallway-continuing-v0', 'POMDP-hallway2-continuing-v0'], [int(2e6), int(5e6)]))
# Common directory names
env_names = ["{}".format(*v) for v in ENVS]
env_name = ["{}".format(*v) for v in ENV]
# rnn_names = ["{}".format(*v) for v in RNN]
rnn_size_names = ["RNNH_{}".format(*v) for v in RNN_SIZE]
rnn_place_names = ["{}".format(v) for v in ['Before', 'After']]
shared_proc_names = ["{}".format(v) for v in ['Unshared', 'Shared']]
delib_names = ["DELIB_{}".format(*v) for v in OC_DELIB]
obs_names = ["OMNI_{}".format(*v) for v in FOMDP]
int_names = ["INT_{}".format(*v) for v in INTEREST]
nopt_names = ["NOPT_{}".format(*v) for v in NUM_OPTIONS]
single_rnn_key = shared_proc_key + rnn_place_key
single_rnn = list(zip([False, True, True], [0, 0, 1]))
single_rnn_names = ['Unshared0','Shared0', 'Shared1']

# A2C
# experiment_title = "A2C_Pomdp"
# variant_levels = list()
# # variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
# # variant_levels.append(VariantLevel(batch_T_key, B_T, ["{}".format(*v) for v in B_T]))  # pomdps
# # variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# # variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
# # variant_levels.append(VariantLevel(shared_proc_key, SHARED_PROC, shared_proc_names))  # Sizes of recurrency
# # variant_levels.append(VariantLevel(rnn_place_key, RNN_PLACE, rnn_place_names))  # Sizes of recurrency
# # variant_levels.append(VariantLevel(rnn_size_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
# # variant_levels.append(VariantLevel(layer_norm_key, LAYER_NORM, ['rnn_norm', 'no_norm']))  # Sizes of recurrency
# # variant_levels.append(VariantLevel(lr_key, lrs, [str(*v) for v in lrs]))  # Learning rates
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
experiment_title = "A2CRnn_Pomdp"
variant_levels = list()
# variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(batch_T_key, B_T, ["{}".format(*v) for v in B_T]))  # pomdps
# variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
variant_levels.append(VariantLevel(single_rnn_key, [single_rnn[-1]], [single_rnn_names[-1]]))  # Shared processor or unshared 1 rnn
variant_levels.append(VariantLevel(pa_key, pas_s, pa_names_s))  # Rnn Placement
variant_levels.append(VariantLevel(pr_key, prs_s, pr_names_s))  # Rnn Placement
# variant_levels.append(VariantLevel(rnn_size_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
# variant_levels.append(VariantLevel(layer_norm_key, LAYER_NORM, ['rnn_norm', 'no_norm']))  # Sizes of recurrency
# variant_levels.append(VariantLevel(lr_key, lrs, [str(*v) for v in lrs]))  # Learning rates
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
# For unshared
experiment_title = "A2CRnn_Pomdp"
variant_levels = list()
# variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
variant_levels.append(VariantLevel(batch_T_key, B_T, ["{}".format(*v) for v in B_T]))  # pomdps
# variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# variant_levels.append(VariantLevel(rnn_type_key, RNN, rnn_names))  # Types of recurrency
variant_levels.append(VariantLevel(shared_proc_key, [(False,)], ['Unshared']))  # Shared processor
variant_levels.append(VariantLevel(rnn_place_key, [(1,)], ['After']))  # Rnn Placement
variant_levels.append(VariantLevel(pa_key, pas, pa_names))  # Rnn Placement
variant_levels.append(VariantLevel(pr_key, prs, pr_names))  # Rnn Placement
# variant_levels.append(VariantLevel(rnn_size_key, RNN_SIZE, rnn_size_names))  # Sizes of recurrency
# variant_levels.append(VariantLevel(layer_norm_key, LAYER_NORM, ['rnn_norm', 'no_norm']))  # Sizes of recurrency
# variant_levels.append(VariantLevel(lr_key, lrs, [str(*v) for v in lrs]))  # Learning rates
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

# A2OC
# experiment_title = "A2OC_Pomdp"
# variant_levels = list()
# # variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
# # variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Option deliberation cost
# # variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
# # variant_levels.append(VariantLevel(shared_proc_key, SHARED_PROC, shared_proc_names))  # Use of interest function
# variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
# # variant_levels.append(VariantLevel(tlr_key, tlrs, [str(*v) for v in tlrs]))  # Termination learning rate
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

use_all = 'All'
use_none = 'None'
use_pi = 'Pi'
use_beta = 'Beta'
use_q = 'Q'
use_pio = 'PIo'
use_pi_all = 'PiPIo'
use_int = 'Int'
oc_prs = list(zip([use_none, use_all]))
oc_prs_unshared = list(zip([use_none, use_all, use_q]))
oc_pr_names = ['NoReward', 'Reward']
oc_pr_unshared_names = ['NoReward', 'Reward', 'QReward']
oc_pas = list(zip([use_none, use_all]))
oc_pa_names = ['NoAction', 'Action']
oc_pas_unshared = list(zip([use_none, use_all, use_pi_all, use_pi]))
oc_pa_unshared_names = ['NoAction', 'AllAction', 'AllPolicyAction', 'IntraPolicyAction']
oc_pos = list(zip([use_none, use_all]))
oc_po_names = ['NoOption', 'Option']
oc_pos_unshared = list(zip([use_none, use_all, use_pi_all, use_pio]))
oc_po_unshared_names = ['NoOption', 'AllOption', 'AllPolicyOption', 'InterPolicyOption']

# A2OC RNN (shared processor)
# experiment_title = "A2OCRnn_Pomdp"
# variant_levels = list()
# # variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
# # variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# # variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Option deliberation cost
# # variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
# variant_levels.append(VariantLevel(single_rnn_key, single_rnn, single_rnn_names))  # Shared processor
# # variant_levels.append(VariantLevel(rnn_place_key, RNN_PLACE, rnn_place_names))
# variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
# variant_levels.append(VariantLevel(pa_key, oc_pas, oc_pa_names))  # Rnn Placement
# variant_levels.append(VariantLevel(pr_key, oc_prs, oc_pr_names))  # Rnn Placement
# variant_levels.append(VariantLevel(po_key, oc_pos, oc_po_names))  # Rnn Placement
# # variant_levels.append(VariantLevel(tlr_key, tlrs, [str(*v) for v in tlrs]))  # Termination learning rate
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_a2oc_rnn,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(oc_key_rnn,),
# )
#
# # A2OC RNN (unshared processor)
# experiment_title = "A2OCRnn_Pomdp"
# variant_levels = list()
# # variant_levels.append(VariantLevel(B_T_env_key, ENVS_PLUS_B_T, env_names))  # pomdps
# variant_levels.append(VariantLevel(envs_plus_params_key, ENVS_PLUS_PARAMS, env_names))  # pomdps
# # variant_levels.append(VariantLevel(fomdp_key, FOMDP, obs_names))  # full or partial observability
# # variant_levels.append(VariantLevel(delib_key, OC_DELIB, delib_names))  # Option deliberation cost
# # variant_levels.append(VariantLevel(nopt_key, NUM_OPTIONS, nopt_names))  # Number of options
# variant_levels.append(VariantLevel(shared_proc_key, [SHARED_PROC[0]], [shared_proc_names[0]]))  # Unshared processor
# variant_levels.append(VariantLevel(rnn_place_key, [RNN_PLACE[1]], [rnn_place_names[1]]))  # Rnn after
# variant_levels.append(VariantLevel(interest_key, INTEREST, int_names))  # Use of interest function
# variant_levels.append(VariantLevel(pa_key, oc_pas_unshared, oc_pa_unshared_names))  # Rnn Placement
# variant_levels.append(VariantLevel(pr_key, oc_prs_unshared, oc_pr_unshared_names))  # Rnn Placement
# variant_levels.append(VariantLevel(po_key, oc_pos_unshared, oc_po_unshared_names))  # Rnn Placement
# # variant_levels.append(VariantLevel(tlr_key, tlrs, [str(*v) for v in tlrs]))  # Termination learning rate
# variants, log_dirs = make_variants(*variant_levels)
# run_experiments(
#     script=path_a2oc_rnn,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(oc_key_rnn,),
# )
