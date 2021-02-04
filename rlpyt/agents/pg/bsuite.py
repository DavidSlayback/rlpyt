from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)
from rlpyt.agents.pg.oc import (CategoricalOcAgent, AlternatingCategoricalOcAgent, RecurrentCategoricalOcAgent, AlternatingRecurrentCategoricalOcAgent)
from rlpyt.models.pg.bsuite_models import BsuiteFfModel, BsuiteRnnModel, BsuiteOcFfModel, BsuiteOcRnnModel

class BsuiteMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract num classes and action size."""
        return dict(input_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)

# Base feedforward
class BsuiteFfAgent(BsuiteMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=BsuiteFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


# Feedforward OC
class BsuiteOcFfAgent(BsuiteMixin, CategoricalOcAgent):
    def __init__(self, ModelCls=BsuiteOcFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingBsuiteOcFfAgent(BsuiteMixin, AlternatingCategoricalOcAgent):
    def __init__(self, ModelCls=BsuiteOcFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


# Base Recurrent
class BsuiteRnnAgent(BsuiteMixin, RecurrentCategoricalPgAgent):
    def __init__(self, ModelCls=BsuiteRnnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingBsuiteRnnAgent(BsuiteMixin, AlternatingRecurrentCategoricalPgAgent):
    def __init__(self, ModelCls=BsuiteRnnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


# Recurrent OC
class BsuiteOcRnnAgent(BsuiteMixin, RecurrentCategoricalOcAgent):
    def __init__(self, ModelCls=BsuiteOcRnnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingBsuiteOcRnnAgent(BsuiteMixin, AlternatingRecurrentCategoricalOcAgent):
    def __init__(self, ModelCls=BsuiteOcRnnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
