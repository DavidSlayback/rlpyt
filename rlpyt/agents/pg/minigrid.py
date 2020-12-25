from rlpyt.agents.pg.categorical import CategoricalPgAgent, RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent
from rlpyt.models.pg.minigrid_ff_model import MinigridFfModel
from rlpyt.models.pg.minigrid_gru_model import MinigridGRUModel

class MinigridMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract image shape and action size."""
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class MinigridFfAgent(MinigridMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=MinigridFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class MinigridGruAgent(MinigridMixin, RecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=MinigridGRUModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingMinigridGruAgent(MinigridMixin,
        AlternatingRecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=MinigridGRUModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)