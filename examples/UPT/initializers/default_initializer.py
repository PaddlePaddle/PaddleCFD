from .base.initializer_base import InitializerBase


class DefaultInitializer(InitializerBase):
    """
    implicitly applies the paddle default initialization
    useful e.g. when defining a list of initializers to sweep over
    """

    def init_weights(self, model):
        pass
