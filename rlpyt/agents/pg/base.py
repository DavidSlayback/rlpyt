
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn",
    ["dist_info", "value", "prev_rnn_state"])

# Option critic agents
# q in addition to value (values of all options vs value of selected option)
# Don't need termination probabilities in step, just terminations (for choosing new options)
# inter-option distribution info (for termination advantage, to weight values of all options)
# prev_o (for termination advantage q_prev_o)
# o (for advantage q_o)
AgentInfoOC = namedarraytuple("AgentInfo", ["dist_info", "q", "value", "termination", "inter_option_dist_info", "prev_o", "o"])  # For option critic agents
AgentInfoOCRnn = namedarraytuple("AgentInfo", ["dist_info", "q", "value", "termination", "inter_option_dist_info", "prev_o", "o", "prev_rnn_state"])  # For option critic agents

