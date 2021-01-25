from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)
from rlpyt.agents.pg.oc import (CategoricalOcAgent, AlternatingCategoricalOcAgent, RecurrentCategoricalOcAgent, AlternatingRecurrentCategoricalOcAgent)
from rlpyt.models.pg.pomdp_models import POMDPFfModel, POMDPRnnModel, POMDPOcModel

class PomdpMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract num classes and action size."""
        return dict(input_classes=env_spaces.observation.n,
                    output_size=env_spaces.action.n)

class PomdpFfAgent(PomdpMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=POMDPFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class PomdpFfOcAgent(PomdpMixin, CategoricalOcAgent):
    def __init__(self, ModelCls=POMDPOcModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class AlternatingPomdpFfOcAgent(PomdpMixin, AlternatingCategoricalOcAgent):
    def __init__(self, ModelCls=POMDPOcModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class PomdpLstmAgent(PomdpMixin, RecurrentCategoricalOcAgent):
    def __init__(self, ModelCls=POMDPRnnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)