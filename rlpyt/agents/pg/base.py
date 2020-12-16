
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn",
    ["dist_info", "value", "prev_rnn_state"])

OCAgentInfo = namedarraytuple("AgentInfo", ["dist_info", "q", "beta", "inter_option_dist_info"])  # For option critic agents
OCAgentInfoRnn = namedarraytuple("AgentInfo", ["dist_info", "q", "beta", "inter_option_dist_info", "prev_rnn_state"])  # For option critic agents

