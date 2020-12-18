
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn",
    ["dist_info", "value", "prev_rnn_state"])

# Option critic agents
# q in addition to value (use value to save average value for all options)
# Don't need termination probabilities in step, just terminations (for choosing new options)
# inter-option distribution info (for termination advantage, to weight values of all options)
# prev_o (for termination advantage q_prev_o)
# o (for advantage q_o)
AgentInfoOC = namedarraytuple("AgentInfoOC", ["dist_info", "dist_info_o", "q", "value", "termination", "dist_info_omega", "prev_o", "o"])  # For option critic agents
AgentInfoOCRnn = namedarraytuple("AgentInfoOCRnn", ["dist_info", "dist_info_o", "q", "value", "termination", "dist_info_omega", "prev_o", "o", "prev_rnn_state"])  # For option critic agents

