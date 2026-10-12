import logging

from providers.path_provider import PathProvider


class InitializerBase:
    def __init__(self, path_provider: PathProvider = None):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider
        assert type(self).get_model_kwargs == InitializerBase.get_model_kwargs

    def init_weights(self, model):
        raise NotImplementedError

    def init_optim(self, model):
        pass

    def get_model_kwargs(self):
        kwargs = self._get_model_kwargs()
        kwargs.pop("is_frozen", None)
        kwargs.pop("freezers", None)
        kwargs.pop("initializers", None)
        kwargs.pop("extractors", None)
        self.logger.info(f"loaded model kwargs: {kwargs}")
        return kwargs

    def _get_model_kwargs(self):
        return {}
