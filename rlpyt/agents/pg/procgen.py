from rlpyt.agents.pg.categorical import CategoricalPgAgent, RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent
from rlpyt.agents.pg.oc import CategoricalOcAgent, AlternatingCategoricalOcAgent, AlternatingRecurrentCategoricalOcAgent, RecurrentCategoricalOcAgent
from rlpyt.models.pg.procgen_ff_model import ProcgenFfModel, ProcgenOcModel

class ProcgenMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract image shape and action size."""
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

class ProcgenFfAgent(ProcgenMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=ProcgenFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class ProcgenOcAgent(ProcgenMixin, CategoricalOcAgent):
    def __init__(self, ModelCls=ProcgenOcModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class AlternatingProcgenOcAgent(ProcgenMixin, AlternatingCategoricalOcAgent):
    def __init__(self, ModelCls=ProcgenOcModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

"""
class MiniWorldLstmAgent(MiniWorldMixin, RecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=MiniWorldLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingMiniWorldLstmAgent(MiniWorldMixin,
        AlternatingRecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=MiniWorldLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
"""